"""
services/evaluation.py — All ESG evaluation business logic.

Helpers for PDF text extraction, AI indicator evaluation, batching, caching,
and score summarisation.  Routes import from here; no FastAPI dependencies.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import logging
import fitz
import google.generativeai as genai
import PyPDF2
from PIL import Image

from infrastructure.s3 import get_pdf_from_s3
from config import BATCH_SIZE, CONCURRENCY_LIMIT, openai_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory text cache (avoids redundant S3 downloads + OCR within a session)
# ---------------------------------------------------------------------------
TEXT_CACHE: Dict[str, dict] = {}
TEXT_CACHE_TTL = 600  # seconds
cache_locks: Dict[str, asyncio.Lock] = {}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def chunk_dict(d: dict, size: int):
    """Yield successive chunks of *size* from dict *d*."""
    items = list(d.items())
    for i in range(0, len(items), size):
        yield dict(items[i:i + size])


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_batched_prompt(indicator_batch: dict, text: str) -> Tuple[str, str]:
    indicators_text = ""

    for code, indicator in indicator_batch.items():
        indicators_text += f"""
Indicator Code: {code}
Title: {indicator['disclosure']}
Description: {indicator['description']}

Scoring Criteria:
0: {indicator['criteria'].get('0', '')}
1: {indicator['criteria'].get('1', '')}
2: {indicator['criteria'].get('2', '')}
3: {indicator['criteria'].get('3', '')}
4: {indicator['criteria'].get('4', '')}

Relevant Keywords: {", ".join(indicator.get("keywords", []))}
---
"""

    system_prompt = f"""
You are an expert ESG analyst using GRI standards.

Evaluate multiple ESG indicators based on the document text.

Rules:
- Assign score 0–4 STRICTLY based on criteria
- Use ONLY information present in the text
- If missing → give lower score
- Do NOT hallucinate
- Provide a detailed explanation (3-5 sentences) of why this score was given
- You MUST reference specific quotes or parts of the text to justify your score
- Evaluate each indicator independently

Return ONLY valid JSON:

{{
  "results": [
    {{
      "indicator_code": "string",
      "score": 0,
      "reasoning": "detailed explanation referencing the text"
    }}
  ]
}}

Indicators:
{indicators_text}
"""

    user_prompt = f"""
DOCUMENT TEXT:
{text[:200000]}
"""

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

async def evaluate_indicator_batch(
    client, text: str, indicator_batch: dict
) -> Tuple[list, dict]:
    system_prompt, user_prompt = build_batched_prompt(indicator_batch, text)

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    token_count = {
        "total_tokens": response.usage.total_tokens,
        "prompt_tokens": response.usage.prompt_tokens,
        "response_tokens": response.usage.completion_tokens,
    }

    try:
        parsed = json.loads(content)
        results = parsed.get("results", [])

        # Robustly handle different keys the AI might use and empty strings
        for item in results:
            reasoning = item.get("reasoning", "").strip()
            if not reasoning:
                reasoning = item.get("explanation", "").strip()
            if not reasoning:
                reasoning = item.get("reason", "").strip()

            if not reasoning:
                item["reasoning"] = "No explanation provided by AI model."
            else:
                item["reasoning"] = reasoning

        return results, token_count
    except Exception as e:
        logging.error(f"Failed to parse JSON from AI: {e}\nContent: {content}")
        return [], token_count


async def evaluate_all_indicators(
    client, extracted_text: str, scoring_rules: dict
) -> dict:
    batches = list(chunk_dict(scoring_rules, BATCH_SIZE))
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def process_batch(batch):
        async with semaphore:
            res, _ = await evaluate_indicator_batch(client, extracted_text, batch)
            return res

    tasks = [process_batch(batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks)

    final_results: dict = {}

    for batch in batch_results:
        for item in batch:
            code = item.get("indicator_code")

            if code in scoring_rules:
                indicator = scoring_rules[code]

                final_results[code] = {
                    "score": item.get("score", 0),
                    "reasoning": item.get("reasoning", ""),
                    "title": indicator["disclosure"],
                    "type": indicator.get("types"),
                    "sub_type": indicator.get("sub-title"),
                    "description": indicator["description"],
                }

    return final_results


# ---------------------------------------------------------------------------
# Single-indicator evaluation (legacy / standalone use)
# ---------------------------------------------------------------------------

async def evaluate_indicator(
    text: str,
    indicator_code: str,
    indicator: dict,
    model_name: str = "gpt-4o-mini",
):
    """
    Evaluate a specific ESG indicator using OpenAI Chat Completions API with
    keyword-based context extraction.  Returns (score, reasoning, token_count).
    """
    keywords = indicator["keywords"]
    relevant_sections: list = []
    text_lower = text.lower()

    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            index = text_lower.find(keyword_lower)
            start = max(0, index - 2000)
            end = min(len(text), index + 2000)
            relevant_sections.append(text[start:end])

            if len(text) > 10000:
                second_index = text_lower.find(keyword_lower, index + 100)
                if second_index > -1 and second_index != index:
                    start2 = max(0, second_index - 2000)
                    end2 = min(len(text), second_index + 2000)
                    relevant_sections.append(text[start2:end2])

    if not relevant_sections:
        relevant_sections = [text[:8000]]

    unique_sections: list = []
    for section in relevant_sections[:4]:
        if section not in unique_sections:
            unique_sections.append(section)

    combined_text = "\n\n[...]\n\n".join(unique_sections)

    if len(combined_text) > 8000:
        combined_text = combined_text[:8000]

    system_prompt = f"""You are an ESG (Environmental, Social, Governance) scoring expert. 
Analyze the following sustainability report extract against the indicator: {indicator_code} - {indicator['disclosure']}.

Indicator description: {indicator['description']}

Relevant keywords to look for: {', '.join(indicator['keywords'])}

Scoring criteria:
0: {indicator['criteria']['0']}
1: {indicator['criteria'].get('1', 'Not specified')}
2: {indicator['criteria'].get('2', 'Not specified')}
3: {indicator['criteria'].get('3', 'Not specified')}
4: {indicator['criteria']['4']}
"""

    # Add reference examples if available
    if "references" in indicator:
        system_prompt += "\n\nREFERENCE EXAMPLES FOR EACH SCORE LEVEL:\n"

        for score in sorted([s for s in indicator["references"].keys() if s.isdigit()]):
            if indicator["references"].get(score):
                system_prompt += f"\n--- EXAMPLE FOR SCORE {score} ---\n"
                system_prompt += f"{indicator['references'][score]}\n"

    # Add instructions and text to analyze
    system_prompt += """
Based on the examples and scoring criteria above, assign a score from 0-4 to the following text.

First give your score as a single digit (0-4), then on a new line provide your explanation.
"""

    user_prompt = f"""
TEXT TO ANALYZE:
{combined_text}
"""

    try:
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        response_text = response.choices[0].message.content.strip()
        lines = response_text.split("\n", 1)

        score = 0
        for char in lines[0].strip():
            if char.isdigit() and int(char) in [0, 1, 2, 3, 4]:
                score = int(char)
                break

        reasoning = lines[1].strip() if len(lines) > 1 else "No explanation provided."

        token_count = {
            "total_tokens": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "response_tokens": response.usage.completion_tokens,
        }

        logger.info(
            f"OpenAI Response for {indicator_code} ({model_name}):\n"
            f"Score: {score}\nReasoning: {reasoning}\nToken Usage: {token_count}"
        )

        return score, reasoning, token_count

    except Exception as e:
        logger.error(f"Error in OpenAI evaluation for {indicator_code}: {str(e)}")
        return 0, f"Error: {str(e)}", {"total_tokens": 0, "prompt_tokens": 0, "response_tokens": 0}


# ---------------------------------------------------------------------------
# Cached text extraction
# ---------------------------------------------------------------------------

async def get_cached_extracted_text(s3_key: str):
    now = time.time()

    # Cleanup expired entries
    for k in list(TEXT_CACHE.keys()):
        if now - TEXT_CACHE[k]["time"] > TEXT_CACHE_TTL:
            del TEXT_CACHE[k]
            if k in cache_locks:
                del cache_locks[k]

    if s3_key not in cache_locks:
        cache_locks[s3_key] = asyncio.Lock()

    async with cache_locks[s3_key]:
        if s3_key in TEXT_CACHE:
            logger.info(f"Using cached OCR text for {s3_key}")
            return TEXT_CACHE[s3_key]["text"], TEXT_CACHE[s3_key]["content"]

        logger.info(f"Downloading and extracting {s3_key}")
        pdf_content = await get_pdf_from_s3(s3_key)
        if not pdf_content:
            return None, None

        extracted_text = await extract_pdf_text(pdf_content)
        TEXT_CACHE[s3_key] = {
            "text": extracted_text,
            "content": pdf_content,
            "time": now,
        }
        return extracted_text, pdf_content


# ---------------------------------------------------------------------------
# Multi-document context builder
# ---------------------------------------------------------------------------

def build_indicator_context(documents: list, indicator: dict) -> dict:
    """
    Select the most appropriate document(s) for a given indicator type and
    return a context object with combined text and source metadata.
    """
    indicator_type = indicator["types"]

    context: dict = {"combined_text": "", "source_documents": []}

    preference_map = {
        "governance": ["sustainability_report", "annual_report", "financial_statement"],
        "environmental": ["sustainability_report", "annual_report"],
        "social": ["sustainability_report", "annual_report"],
        "economic": ["financial_statement", "annual_report", "sustainability_report"],
    }

    preferences = preference_map.get(
        indicator_type,
        ["sustainability_report", "annual_report", "financial_statement"],
    )

    primary_docs = [doc for doc in documents if doc["type"] in preferences[:1]]

    if not primary_docs:
        for pref in preferences:
            docs_of_type = [doc for doc in documents if doc["type"] == pref]
            if docs_of_type:
                primary_docs = docs_of_type
                break

    if not primary_docs and documents:
        primary_docs = [documents[0]]

    all_text = ""
    for doc in primary_docs:
        context["source_documents"].append(doc["filename"])
        all_text += doc["text"] + "\n\n"

    context["combined_text"] = all_text
    return context


async def evaluate_indicator_with_context(
    context: dict,
    indicator_code: str,
    indicator: dict,
    model_name: str = "gpt-4o-mini",
):
    """
    Evaluate an indicator using context built from multiple documents.
    Returns (score, reasoning, token_count).
    """
    combined_text = context["combined_text"]

    if not combined_text.strip():
        return (
            0,
            "No relevant text found in provided documents.",
            {"total_tokens": 0, "prompt_tokens": 0, "response_tokens": 0},
        )

    keywords = indicator["keywords"]
    relevant_sections: list = []
    text_lower = combined_text.lower()

    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            index = text_lower.find(keyword_lower)
            start = max(0, index - 2000)
            end = min(len(combined_text), index + 2000)
            relevant_sections.append(combined_text[start:end])

            if len(combined_text) > 10000:
                second_index = text_lower.find(keyword_lower, index + 100)
                if second_index > -1 and second_index != index:
                    start2 = max(0, second_index - 2000)
                    end2 = min(len(combined_text), second_index + 2000)
                    relevant_sections.append(combined_text[start2:end2])

    if not relevant_sections:
        relevant_sections = [combined_text[:8000]]

    unique_sections: list = []
    for section in relevant_sections[:4]:
        if section not in unique_sections:
            unique_sections.append(section)

    combined_text = "\n\n[...]\n\n".join(unique_sections)

    if len(combined_text) > 8000:
        combined_text = combined_text[:8000]

    system_prompt = f"""You are an ESG (Environmental, Social, Governance) scoring expert. 
Analyze the following extracted text against the indicator: {indicator_code} - {indicator['disclosure']}.

Indicator description: {indicator['description']}

Relevant keywords to look for: {', '.join(indicator['keywords'])}

Scoring criteria:
0: {indicator['criteria']['0']}
1: {indicator['criteria'].get('1', 'Not specified')}
2: {indicator['criteria'].get('2', 'Not specified')}
3: {indicator['criteria'].get('3', 'Not specified')}
4: {indicator['criteria']['4']}

Based on the text provided by the user, assign a score from 0-4 and provide a brief explanation (2-3 sentences) justifying your score.

First give your score as a single digit (0-4), then on a new line provide your explanation.

Example:
3
The text clearly describes procedures for waste management including recycling programs. It provides quantitative data on waste reduction but lacks complete information on circular economy implementation.
"""

    user_prompt = f"""
TEXT TO ANALYZE:
{combined_text}
"""

    try:
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        response_text = response.choices[0].message.content.strip()
        lines = response_text.split("\n", 1)

        score = 0
        for char in lines[0].strip():
            if char.isdigit() and int(char) in [0, 1, 2, 3, 4]:
                score = int(char)
                break

        reasoning = lines[1].strip() if len(lines) > 1 else "No explanation provided."

        token_count = {
            "total_tokens": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "response_tokens": response.usage.completion_tokens,
        }

        return score, reasoning, token_count

    except Exception as e:
        return 0, f"Error: {str(e)}", {"total_tokens": 0, "prompt_tokens": 0, "response_tokens": 0}


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

def calculate_summary_scores(results: dict) -> dict:
    """Calculate summary scores by category."""
    total_score = sum(result["score"] for result in results.values())
    return {"spdi_index": total_score}


# ---------------------------------------------------------------------------
# PDF text extraction (PyPDF2 → Gemini OCR → OpenAI vision fallback)
# ---------------------------------------------------------------------------

async def extract_pdf_text(pdf_content: bytes) -> str:
    """Extract text from PDF with fallback to Gemini for scanned documents."""
    pdf_file = io.BytesIO(pdf_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    extracted_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text() or ""
        extracted_text += page_text + "\n\n"

    avg_chars_per_page = (
        len(extracted_text) / len(pdf_reader.pages) if len(pdf_reader.pages) > 0 else 0
    )

    if avg_chars_per_page < 200:
        logger.info(
            f"Detected potential scanned PDF (avg {avg_chars_per_page:.2f} chars/page). "
            "Using Gemini for image processing..."
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(pdf_content)
            temp_pdf_path = temp_pdf.name

        try:
            doc = fitz.open(temp_pdf_path)
            gemini_text = ""

            model = genai.GenerativeModel("gemini-2.5-flash")

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")

                try:
                    await asyncio.sleep(1)  # Rate limiting
                    response = await model.generate_content_async(
                        [
                            "Extract all text visible in this image. Format as plain text with paragraphs preserved.",
                            {"mime_type": "image/png", "data": img_base64},
                        ]
                    )
                    gemini_text += response.text + "\n\n"
                except Exception as e:
                    logger.warning(
                        f"Gemini OCR failed for page {page_num + 1} ({e}). "
                        "Falling back to OpenAI..."
                    )
                    try:
                        response = await openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Extract all text visible in this image. Format as plain text with paragraphs preserved.",
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{img_base64}"
                                            },
                                        },
                                    ],
                                }
                            ],
                            max_tokens=2000,
                        )
                        gemini_text += response.choices[0].message.content + "\n\n"
                    except Exception as openai_e:
                        logger.error(f"Error processing page {page_num + 1} with OpenAI fallback: {openai_e}")

            doc.close()

            if len(gemini_text.strip()) > len(extracted_text.strip()):
                return gemini_text
        except Exception as e:
            logger.error(f"Error in image-based extraction: {e}")
        finally:
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)

    return extracted_text


def check_extraction_quality(extracted_text: str, pdf_reader) -> dict:
    """
    Check the quality of PDF extraction and return diagnostics information.
    """
    total_pages = len(pdf_reader.pages)
    char_count = len(extracted_text)
    words = extracted_text.split()
    word_count = len(words)

    avg_chars_per_page = char_count / total_pages if total_pages > 0 else 0

    extraction_issues: list = []

    if char_count == 0:
        extraction_issues.append("No text extracted from the PDF")
    elif avg_chars_per_page < 200:
        extraction_issues.append("Very little text extracted per page, possible scanned PDF")

    common_esg_terms = [
        "sustainability", "environmental", "social", "governance", "report",
        "energy", "waste", "emissions", "water", "compliance", "policy",
    ]

    found_terms = [term for term in common_esg_terms if term.lower() in extracted_text.lower()]
    term_coverage = len(found_terms) / len(common_esg_terms)

    if term_coverage < 0.2:
        extraction_issues.append(
            "Few sustainability terms found, possible extraction issue or wrong document type"
        )

    return {
        "total_pages": total_pages,
        "characters_extracted": char_count,
        "words_extracted": word_count,
        "avg_chars_per_page": round(avg_chars_per_page, 2),
        "esg_terms_found": found_terms,
        "esg_term_coverage": f"{round(term_coverage * 100, 1)}%",
        "extraction_issues": extraction_issues,
        "extraction_success": len(extraction_issues) == 0,
    }
