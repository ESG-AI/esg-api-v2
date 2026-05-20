"""
routes/evaluate.py — ESG evaluation endpoints.

Endpoints:
  POST /extract          Extract text from a PDF (diagnostics only)
  POST /evaluate         Run a synchronous ESG evaluation
  POST /evaluate-multi   Evaluate multiple documents with batching + retry
"""

import asyncio
import io
import time
from typing import Optional

import PyPDF2
from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile

from aws import get_pdf_from_s3
from config import BATCH_SIZE, CONCURRENCY_LIMIT, logger, openai_client, scoring_rules
from schemas import EvaluateMultiRequest, EvaluateRequest
from services.evaluation import (
    calculate_summary_scores,
    check_extraction_quality,
    chunk_dict,
    evaluate_all_indicators,
    evaluate_indicator_batch,
    extract_pdf_text,
    get_cached_extracted_text,
)
from db.repositories.document import save_analysis_results

router = APIRouter(tags=["Evaluate"])


@router.post("/extract")
async def extract_pdf(pdf: UploadFile = File(...)):
    """
    Extract text from a PDF and return detailed extraction diagnostics.
    This endpoint is for testing extraction quality without running the full
    ESG evaluation.
    """
    try:
        pdf_content = await pdf.read()

        extracted_text = await extract_pdf_text(pdf_content)

        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        extraction_quality = check_extraction_quality(extracted_text, pdf_reader)

        avg_chars_per_page = (
            len(extracted_text) / len(pdf_reader.pages) if len(pdf_reader.pages) > 0 else 0
        )
        used_gemini = avg_chars_per_page < 200 and len(extracted_text.strip()) > 0

        page_details = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text() or ""
            page_details.append(
                {
                    "page_number": page_num + 1,
                    "characters": len(page_text),
                    "words": len(page_text.split()) if page_text else 0,
                    "empty": len(page_text.strip()) == 0,
                }
            )

        text_sample = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text

        return {
            "filename": pdf.filename,
            "extraction_quality": extraction_quality,
            "page_details": page_details,
            "text_sample": text_sample,
            "text_length": len(extracted_text),
            "page_count": len(pdf_reader.pages),
            "empty_pages": sum(1 for page in page_details if page["empty"]),
            "used_gemini_ocr": used_gemini,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting PDF: {str(e)}")


@router.post("/evaluate")
async def evaluate_pdf(request: EvaluateRequest):
    start_time = time.time()

    if not request.s3_object_key:
        raise HTTPException(status_code=400, detail="s3_object_key is required")

    # 1. Download PDF from S3
    pdf_content = await get_pdf_from_s3(request.s3_object_key)
    if not pdf_content:
        raise HTTPException(status_code=404, detail="Failed to retrieve document from S3")

    # 2. Extract text
    extracted_text = await extract_pdf_text(pdf_content)

    # 3. Evaluate indicators
    results = await evaluate_all_indicators(openai_client, extracted_text, scoring_rules)

    # 4. Compute total score
    total_score = sum(item.get("score", 0) for item in results.values())

    return {
        "results": results,
        "total_score": total_score,
        "processing_time": round(time.time() - start_time, 2),
    }


@router.post("/evaluate-multi")
async def evaluate_multi_documents(
    request: EvaluateMultiRequest = Body(...),
    gri_type: Optional[str] = Query(
        None, description="One of: governance, economic, social, environmental"
    ),
):
    """
    Process multiple documents using existing S3 object keys with
    client-specified document types and GRI type filtering.

    Each document type should be one of:
      'sustainability_report', 'annual_report', 'financial_statement'
    If document_types is not provided, all documents will be treated as
    'sustainability_report'.
    If gri_type is provided, only indicators of that type will be evaluated.
    """
    # Validate document type
    if request.document_types:
        for doc_type in request.document_types:
            if doc_type and doc_type not in [
                "sustainability_report",
                "annual_report",
                "financial_statement",
            ]:
                raise HTTPException(status_code=400, detail=f"Invalid document type: {doc_type}")

    # Filter indices by gri_type if provided
    if gri_type:
        indices_to_process = [
            code for code, rule in scoring_rules.items() if rule["types"] == gri_type
        ]
    else:
        indices_to_process = list(scoring_rules.keys())

    start_time = time.time()

    try:
        documents = []
        file_details = []

        extraction_start = time.time()

        for i, s3_object_key in enumerate(request.s3_object_keys):
            extracted_text, pdf_content = await get_cached_extracted_text(s3_object_key)

            if not pdf_content:
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to retrieve document from S3: {s3_object_key}",
                )

            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            extraction_quality = check_extraction_quality(extracted_text, pdf_reader)

            doc_type = "sustainability_report"
            if (
                request.document_types
                and i < len(request.document_types)
                and request.document_types[i]
            ):
                if request.document_types[i] in [
                    "sustainability_report",
                    "annual_report",
                    "financial_statement",
                ]:
                    doc_type = request.document_types[i]

            filename = s3_object_key
            if request.filenames and i < len(request.filenames) and request.filenames[i]:
                filename = request.filenames[i]

            file_details.append(
                {
                    "filename": filename,
                    "s3_object_key": s3_object_key,
                    "file_size": len(pdf_content),
                    "extraction_quality": extraction_quality,
                    "document_type": doc_type,
                }
            )

            documents.append({"filename": filename, "text": extracted_text, "type": doc_type})

        extraction_time = time.time() - extraction_start

        total_tokens_used = 0
        token_usage_by_indicator: dict = {}
        ai_processing_times: dict = {}
        ai_evaluation_start = time.time()

        results: dict = {}

        full_document_text = "\n\n".join([doc["text"] for doc in documents])

        scoring_rules_to_process = {k: scoring_rules[k] for k in indices_to_process}
        batches = list(chunk_dict(scoring_rules_to_process, BATCH_SIZE))

        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def process_batch(batch):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    batch_start = time.time()
                    async with semaphore:
                        await asyncio.sleep(0.5)
                        batch_results_list, token_count = await evaluate_indicator_batch(
                            openai_client, full_document_text, batch
                        )
                    batch_time = time.time() - batch_start
                    return batch, batch_results_list, token_count, batch_time, None
                except Exception as e:
                    error_message = str(e)
                    if "429" in error_message and attempt < max_retries - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    return (
                        batch,
                        [],
                        {"total_tokens": 0, "prompt_tokens": 0, "response_tokens": 0},
                        0,
                        error_message,
                    )
            return (
                batch,
                [],
                {"total_tokens": 0, "prompt_tokens": 0, "response_tokens": 0},
                0,
                "Max retries exceeded",
            )

        tasks = [process_batch(b) for b in batches]
        batch_results = await asyncio.gather(*tasks)

        for batch_dict, res_list, t_count, b_time, err in batch_results:
            if t_count.get("total_tokens"):
                total_tokens_used += t_count["total_tokens"]

            ai_lookup = {
                item.get("indicator_code"): item
                for item in res_list
                if isinstance(item, dict)
            }

            for code, indicator in batch_dict.items():
                token_usage_by_indicator[code] = t_count
                ai_processing_times[code] = (
                    round(b_time / len(batch_dict), 2) if len(batch_dict) > 0 else 0
                )

                if err or code not in ai_lookup:
                    if not err:
                        logger.warning(
                            f"Indicator '{code}' was silently dropped from AI batch response "
                            "— will retry."
                        )
                    else:
                        logger.error(f"Batch error for indicator '{code}': {err}")
                    results[code] = {
                        "score": 0,
                        "title": indicator.get("disclosure", "Unknown"),
                        "type": indicator.get("types", "Unknown"),
                        "sub_type": indicator.get("sub-title", "Unknown"),
                        "description": indicator.get("description", ""),
                        "reasoning": (
                            f"Evaluation error: {err}"
                            if err
                            else "Indicator was not returned by AI — pending retry."
                        ),
                        "error": err if err else "Missing from AI response",
                    }
                else:
                    item = ai_lookup[code]
                    results[code] = {
                        "score": item.get("score", 0),
                        "reasoning": item.get("reasoning", ""),
                        "source_documents": [doc["filename"] for doc in documents],
                        "title": indicator["disclosure"],
                        "type": indicator["types"],
                        "sub_type": indicator.get("sub-title", "Unknown"),
                        "description": indicator.get("description", ""),
                        "token_usage": t_count,
                    }

        ai_evaluation_time = time.time() - ai_evaluation_start

        # --- Retry indicators that were dropped or errored in the first pass ---
        missing_indicators = {
            code: scoring_rules_to_process[code]
            for code in scoring_rules_to_process
            if "error" in results.get(code, {})
        }
        if missing_indicators:
            logger.warning(
                f"Retrying {len(missing_indicators)} indicator(s) missing from initial batches: "
                f"{list(missing_indicators.keys())}"
            )
            try:
                retry_results_list, retry_token_count = await evaluate_indicator_batch(
                    openai_client, full_document_text, missing_indicators
                )
                total_tokens_used += retry_token_count.get("total_tokens", 0)

                returned_codes = [
                    item.get("indicator_code")
                    for item in retry_results_list
                    if isinstance(item, dict)
                ]
                logger.info(f"AI returned codes in retry batch: {returned_codes}")

                retry_lookup = {
                    item.get("indicator_code"): item
                    for item in retry_results_list
                    if isinstance(item, dict)
                }

                still_missing = []
                for code, indicator in missing_indicators.items():
                    if code in retry_lookup:
                        retry_item = retry_lookup[code]
                        results[code] = {
                            "score": retry_item.get("score", 0),
                            "reasoning": retry_item.get("reasoning", ""),
                            "source_documents": [doc["filename"] for doc in documents],
                            "title": indicator["disclosure"],
                            "type": indicator["types"],
                            "sub_type": indicator.get("sub-title", "Unknown"),
                            "description": indicator.get("description", ""),
                            "token_usage": retry_token_count,
                            "retried": True,
                        }
                        logger.info(f"Retry batch succeeded for indicator '{code}'.")
                    else:
                        fuzzy_match = next(
                            (
                                ret_code
                                for ret_code in retry_lookup
                                if code.startswith(ret_code) or ret_code.startswith(code)
                            ),
                            None,
                        )
                        if fuzzy_match:
                            retry_item = retry_lookup[fuzzy_match]
                            results[code] = {
                                "score": retry_item.get("score", 0),
                                "reasoning": retry_item.get("reasoning", ""),
                                "source_documents": [doc["filename"] for doc in documents],
                                "title": indicator["disclosure"],
                                "type": indicator["types"],
                                "sub_type": indicator.get("sub-title", "Unknown"),
                                "description": indicator.get("description", ""),
                                "token_usage": retry_token_count,
                                "retried": True,
                            }
                            logger.info(f"Fuzzy match: '{code}' resolved via '{fuzzy_match}'.")
                        else:
                            still_missing.append(code)

                # For indicators still missing, retry each one individually
                if still_missing:
                    logger.warning(
                        f"Retrying {len(still_missing)} indicator(s) one-by-one: {still_missing}"
                    )
                    for code in still_missing:
                        indicator = missing_indicators[code]
                        try:
                            solo_results, solo_tokens = await evaluate_indicator_batch(
                                openai_client, full_document_text, {code: indicator}
                            )
                            total_tokens_used += solo_tokens.get("total_tokens", 0)
                            solo_codes = [
                                item.get("indicator_code")
                                for item in solo_results
                                if isinstance(item, dict)
                            ]
                            logger.info(f"Solo retry for '{code}' — AI returned: {solo_codes}")

                            matched = next(
                                (
                                    item
                                    for item in solo_results
                                    if isinstance(item, dict)
                                    and item.get("indicator_code") == code
                                ),
                                None,
                            )
                            if not matched:
                                matched = next(
                                    (item for item in solo_results if isinstance(item, dict)),
                                    None,
                                )

                            if matched:
                                results[code] = {
                                    "score": matched.get("score", 0),
                                    "reasoning": matched.get("reasoning", ""),
                                    "source_documents": [doc["filename"] for doc in documents],
                                    "title": indicator["disclosure"],
                                    "type": indicator["types"],
                                    "sub_type": indicator.get("sub-title", "Unknown"),
                                    "description": indicator.get("description", ""),
                                    "token_usage": solo_tokens,
                                    "retried": True,
                                }
                                logger.info(f"Solo retry succeeded for indicator '{code}'.")
                            else:
                                results[code]["reasoning"] = (
                                    "Indicator could not be evaluated after multiple retries."
                                )
                                logger.error(f"Indicator '{code}' still missing after solo retry.")
                        except Exception as solo_err:
                            results[code]["reasoning"] = f"Solo retry failed: {solo_err}"
                            logger.error(f"Solo retry failed for '{code}': {solo_err}")

            except Exception as retry_err:
                logger.error(f"Retry batch failed entirely: {retry_err}")
                for code in missing_indicators:
                    results[code]["reasoning"] = f"Retry also failed: {retry_err}"
        # --- End retry ---

        summary = calculate_summary_scores(results)
        total_time = time.time() - start_time

        timing_metrics = {
            "total_processing_time_seconds": round(total_time, 2),
            "s3_upload_time_seconds": 0,
            "extraction_time_seconds": round(extraction_time, 2),
            "extraction_quality_check_time_seconds": 0,
            "ai_evaluation_time_seconds": round(ai_evaluation_time, 2),
            "indicator_processing_times": ai_processing_times,
            "db_save_time_seconds": 0,
        }

        token_usage_data = {
            "total_tokens_used": total_tokens_used,
            "by_indicator": token_usage_by_indicator,
        }

        db_save_start = time.time()
        if file_details:
            document_id = save_analysis_results(
                filename=file_details[0]["filename"],
                s3_object_key=file_details[0]["s3_object_key"],
                file_size=file_details[0]["file_size"],
                extraction_quality=file_details[0]["extraction_quality"],
                results=results,
                summary=summary,
                token_usage=token_usage_by_indicator,
                performance_metrics=timing_metrics,
                user_id=request.user_id,
            )
        else:
            document_id = None

        db_save_time = time.time() - db_save_start
        timing_metrics["db_save_time_seconds"] = round(db_save_time, 2)

        return {
            "id": document_id,
            "documents": [{"filename": doc["filename"], "type": doc["type"]} for doc in documents],
            "gri_type": gri_type,
            "indicators": results,
            "summary": summary,
            "token_usage": token_usage_data,
            "performance_metrics": timing_metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")
