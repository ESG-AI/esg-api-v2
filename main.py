from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body, Depends
import google.generativeai as genai
import fitz
import json
import os
import io
import PyPDF2
import asyncio
from PIL import Image
import base64
import tempfile
import time
from fastapi.middleware.cors import CORSMiddleware
from neon import AnalysisResult, SessionLocal, init_db, save_analysis_results, get_document_analysis, get_all_documents, Document
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
from aws import upload_to_s3, get_pdf_from_s3
import logging
from pydantic import BaseModel


app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow only the deployed frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db_client():
    init_db()

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

print(f"Gemini SDK version: {genai.__version__}")

# Load scoring rules from JSON file
with open("scoring_rules.json", "r") as f:
    scoring_rules = json.load(f)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gemini_prompts.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('gemini_prompts')

@app.post("/extract")
async def extract_pdf(pdf: UploadFile = File(...)):
    """
    Extract text from a PDF and return detailed extraction diagnostics.
    This endpoint is for testing extraction quality without running the full ESG evaluation.
    """
    try:
        pdf_content = await pdf.read()
        
        # Use the enhanced extraction function with Gemini fallback for scanned documents
        extracted_text = await extract_pdf_text(pdf_content)
        
        # For diagnostics, still create a PDF reader
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Get extraction quality stats
        extraction_quality = check_extraction_quality(extracted_text, pdf_reader)
        
        # Check if Gemini was used for extraction
        avg_chars_per_page = len(extracted_text) / len(pdf_reader.pages) if len(pdf_reader.pages) > 0 else 0
        used_gemini = avg_chars_per_page < 200 and len(extracted_text.strip()) > 0
        
        # Track per-page stats (from the original PDF, not the enhanced extraction)
        page_details = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text() or ""
            page_details.append({
                "page_number": page_num + 1,
                "characters": len(page_text),
                "words": len(page_text.split()) if page_text else 0,
                "empty": len(page_text.strip()) == 0
            })
        
        # Get sample text (first 1000 chars)
        text_sample = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
        
        return {
            "filename": pdf.filename,
            "extraction_quality": extraction_quality,
            "page_details": page_details,
            "text_sample": text_sample,
            "text_length": len(extracted_text),
            "page_count": len(pdf_reader.pages),
            "empty_pages": sum(1 for page in page_details if page["empty"]),
            "used_gemini_ocr": used_gemini
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting PDF: {str(e)}")

def check_extraction_quality(extracted_text, pdf_reader):
    """
    Check the quality of PDF extraction and return diagnostics information.
    
    Args:
        extracted_text: The text extracted from the PDF
        pdf_reader: The PyPDF2 reader object with the original PDF
    
    Returns:
        dict: Diagnostics information about the extraction quality
    """
    total_pages = len(pdf_reader.pages)
    char_count = len(extracted_text)
    words = extracted_text.split()
    word_count = len(words)
    
    # Calculate average text per page
    avg_chars_per_page = char_count / total_pages if total_pages > 0 else 0
    
    # Check if text extraction might have failed
    extraction_issues = []
    
    if char_count == 0:
        extraction_issues.append("No text extracted from the PDF")
    elif avg_chars_per_page < 200:  # Arbitrary threshold for minimal text on a page
        extraction_issues.append("Very little text extracted per page, possible scanned PDF")
    
    # Check for common text markers that should exist in sustainability reports
    common_esg_terms = ["sustainability", "environmental", "social", "governance", "report", 
                        "energy", "waste", "emissions", "water", "compliance", "policy"]
    
    found_terms = []
    for term in common_esg_terms:
        if term.lower() in extracted_text.lower():
            found_terms.append(term)
    
    term_coverage = len(found_terms) / len(common_esg_terms)
    
    if term_coverage < 0.2:  # Less than 20% of expected terms found
        extraction_issues.append("Few sustainability terms found, possible extraction issue or wrong document type")
    
    # Generate diagnostics report
    diagnostics = {
        "total_pages": total_pages,
        "characters_extracted": char_count,
        "words_extracted": word_count,
        "avg_chars_per_page": round(avg_chars_per_page, 2),
        "esg_terms_found": found_terms,
        "esg_term_coverage": f"{round(term_coverage * 100, 1)}%",
        "extraction_issues": extraction_issues,
        "extraction_success": len(extraction_issues) == 0
    }
    
    return diagnostics

@app.post("/upload")
async def upload_pdf(pdf: UploadFile = File(...)):
    """
    Upload a PDF to S3 and return the S3 object key.
    """
    try:
        pdf_content = await pdf.read()
        from aws import upload_to_s3
        s3_object_key = await upload_to_s3(pdf_content, pdf.filename)
        if not s3_object_key:
            raise HTTPException(status_code=500, detail="Failed to upload document to S3")
        return {"s3_object_key": s3_object_key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

class EvaluateRequest(BaseModel):
    s3_object_key: Optional[str] = None
    filename: Optional[str] = None

@app.post("/evaluate")
async def evaluate_pdf(
    request: EvaluateRequest = Body(...),
    document_type: str = "sustainability_report",
    gri_type: Optional[str] = Query(None, description="One of: governance, economic, social, environmental")
    ):
    """
    Evaluate a single PDF against all ESG indices with specified document type or only those of a specific GRI type.
    Accepts only an existing S3 object key in the request body.
    """
    # Validate document type
    if document_type not in ["sustainability_report", "annual_report", "financial_statement"]:
        document_type = "sustainability_report"  # Default if invalid type provided
    
    # Filter indices by gri_type if provided
    if gri_type:
        indices_to_process = [code for code, rule in scoring_rules.items() if rule["types"] == gri_type]
    else:
        indices_to_process = list(scoring_rules.keys())
    
    # Track overall processing time
    start_time = time.time()

    try:
        s3_object_key = request.s3_object_key
        filename = request.filename
        if s3_object_key:
            from aws import get_pdf_from_s3
            pdf_content = await get_pdf_from_s3(s3_object_key)
            s3_object_key_final = s3_object_key
            s3_upload_time = 0
        else:
            raise HTTPException(status_code=400, detail="Must provide an S3 object key in the request body")

        # Use provided filename or fallback to s3_object_key
        filename_to_save = filename if filename else s3_object_key_final

        # Extraction timing
        extraction_start = time.time()
        extracted_text = await extract_pdf_text(pdf_content)
        extraction_time = time.time() - extraction_start
        
        # Extraction quality check timing
        quality_check_start = time.time()
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        extraction_quality = check_extraction_quality(extracted_text, pdf_reader)
        quality_check_time = time.time() - quality_check_start
        
        if not extraction_quality["extraction_success"]:
            extraction_warning = f"Warning: PDF extraction issues detected: {', '.join(extraction_quality['extraction_issues'])}"
        else:
            extraction_warning = None

        # Get file size
        file_size = len(pdf_content)
        
        # Track token usage across all evaluations
        total_tokens_used = 0
        token_usage_by_indicator = {}

        # Track AI processing times for each indicator
        ai_processing_times = {}
        ai_evaluation_start = time.time()
        
        # Evaluate only the selected indices
        results = {}
        for indicator_code in indices_to_process:
            indicator = scoring_rules[indicator_code]
            try:
                await asyncio.sleep(1)  # Rate limiting for API calls

                # Track time for this specific indicator
                indicator_start = time.time()
                score, reasoning, token_count = await evaluate_indicator(extracted_text, indicator_code, indicator)
                indicator_time = time.time() - indicator_start
                
                # Store timing information
                ai_processing_times[indicator_code] = round(indicator_time, 2)
                
                # Store token usage information
                token_usage_by_indicator[indicator_code] = token_count
                if token_count["total_tokens"]:
                    total_tokens_used += token_count["total_tokens"]
                
                results[indicator_code] = {
                    "score": score,
                    "reasoning": reasoning,
                    "title": indicator["disclosure"],
                    "type": indicator["types"],
                    "sub_type": indicator["sub-title"],
                    "description": indicator["description"],
                    "token_usage": token_count
                }
            except Exception as e:
                # Handle errors for individual indicators instead of failing the whole process
                error_message = str(e)
                if "429" in error_message:  # Rate limit error
                    # Wait longer if we hit a rate limit
                    await asyncio.sleep(5)
                    # Try again for this indicator
                    continue
                    
                # Log the error but continue with other indicators
                results[indicator_code] = {
                    "score": 0,
                    "title": indicator.get("disclosure", "Unknown"),
                    "type": indicator.get("types", "Unknown"),
                    "sub_type": indicator.get("sub-title", "Unknown"),
                    "description": indicator.get("description", ""),
                    "error": error_message
                }
        ai_evaluation_time = time.time() - ai_evaluation_start
        
        # Calculate summary scores by category
        summary = calculate_summary_scores(results)

        # Calculate total processing time
        total_time = time.time() - start_time

        # Prepare token usage data
        token_usage_data = {
            "total_tokens_used": total_tokens_used,
            "by_indicator": token_usage_by_indicator
        }
        
        # Add timing metrics to response
        timing_metrics = {
            "total_processing_time_seconds": round(total_time, 2),
            "s3_upload_time_seconds": s3_upload_time,
            "extraction_time_seconds": round(extraction_time, 2),
            "extraction_quality_check_time_seconds": round(quality_check_time, 2),
            "ai_evaluation_time_seconds": round(ai_evaluation_time, 2),
            "indicator_processing_times": ai_processing_times
        }

        document_id = save_analysis_results(
            filename=filename_to_save,
            s3_object_key=s3_object_key_final,
            file_size=file_size,
            extraction_quality=extraction_quality,
            results=results,
            summary=summary,
            token_usage=token_usage_by_indicator,
            performance_metrics=timing_metrics
        )
        
        return {
            "id": document_id,
            "filename": filename_to_save,
            "document_type": document_type,
            "gri_type": gri_type,
            "s3_object_key": s3_object_key_final,
            "extraction_quality": extraction_quality,
            "extraction_warning": extraction_warning,
            "indicators": results,
            "summary": summary,
            "token_usage": token_usage_data,
            "performance_metrics": timing_metrics
        }
    except Exception as e:
        # Calculate time even for errors
        error_time = time.time() - start_time
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

async def evaluate_indicator(text, indicator_code, indicator):
    """
    Evaluate a specific ESG indicator using Gemini AI with keyword-based context extraction.
    Finds the most relevant parts of the document by locating ESG index keywords.
    Returns both score and token usage information.
    """
    
    # Configure Gemini model for this specific evaluation
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Find the most relevant sections of text based on keywords
    keywords = indicator['keywords']
    relevant_sections = []
    text_lower = text.lower()
    
    # Try to find sections containing keywords
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            # Find position of keyword
            index = text_lower.find(keyword_lower)
            # Extract surrounding context (2000 chars before and after)
            start = max(0, index - 2000)
            end = min(len(text), index + 2000)
            relevant_sections.append(text[start:end])
            
            # Look for more instances if document is large
            if len(text) > 10000:
                # Find a second instance further in the document
                second_index = text_lower.find(keyword_lower, index + 100)
                if second_index > -1 and second_index != index:
                    start2 = max(0, second_index - 2000)
                    end2 = min(len(text), second_index + 2000)
                    relevant_sections.append(text[start2:end2])
    
    # If no relevant sections with keywords were found, use the beginning of the document
    if not relevant_sections:
        relevant_sections = [text[:8000]]
    
    # Use the most relevant sections, but ensure we don't exceed token limits
    # Priority: take the first 4 unique sections (or fewer if that's all we have)
    unique_sections = []
    for section in relevant_sections[:4]:  # Limit to first 4 sections
        if section not in unique_sections:
            unique_sections.append(section)
    
    # Combine sections with indicators of text breaks
    combined_text = "\n\n[...]\n\n".join(unique_sections)
    
    # Ensure we don't exceed token limits (8000 chars is a safe limit)
    if len(combined_text) > 8000:
        combined_text = combined_text[:8000]
    
    # Update prompt to request both score and reasoning
    prompt = f"""
    You are an ESG (Environmental, Social, Governance) scoring expert. 
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
        prompt += "\n\nREFERENCE EXAMPLES FOR EACH SCORE LEVEL:\n"
        
        for score in sorted([s for s in indicator["references"].keys() if s.isdigit()]):
            if indicator["references"].get(score):
                prompt += f"\n--- EXAMPLE FOR SCORE {score} ---\n"
                prompt += f"{indicator['references'][score]}\n"
    
     # Add instructions and text to analyze
    prompt += f"""
    Based on the examples and scoring criteria above, assign a score from 0-4 to the following text.
    
    First give your score as a single digit (0-4), then on a new line provide your explanation.
    
    TEXT TO ANALYZE:
    {combined_text}
    """

    # Get token count for the prompt before sending
    prompt_token_count = model.count_tokens(prompt).total_tokens
    
    # Send request to Gemini
    try:
        response = await model.generate_content_async(prompt)
        
        # Parse response to extract score and reasoning
        response_text = response.text.strip()
        lines = response_text.split("\n", 1)
        
        # Extract score from first line
        score = 0
        for char in lines[0].strip():
            if char.isdigit() and int(char) in [0, 1, 2, 3, 4]:
                score = int(char)
                break
        
        # Extract reasoning from remaining text
        reasoning = lines[1].strip() if len(lines) > 1 else "No explanation provided."
        
        # Get token usage
        token_count = {
            "total_tokens": prompt_token_count + (len(response_text) // 4),
            "prompt_tokens": prompt_token_count,
            "response_tokens": len(response_text) // 4
        }
        
        # Log the response
        logger.info(f"Gemini Response for {indicator_code}:\nScore: {score}\nReasoning: {reasoning}\nToken Usage: {token_count}")
        
        return score, reasoning, token_count
        
    except Exception as e:
        logger.error(f"Error in Gemini evaluation for {indicator_code}: {str(e)}")
        return 0, f"Error: {str(e)}", {"total_tokens": prompt_token_count, "prompt_tokens": prompt_token_count, "response_tokens": 0}

class EvaluateMultiRequest(BaseModel):
    s3_object_keys: List[str]
    filenames: Optional[List[str]] = None
    document_types: Optional[List[str]] = None

@app.post("/evaluate-multi")
async def evaluate_multi_documents(
    request: EvaluateMultiRequest = Body(...),
    gri_type: Optional[str] = Query(None, description="One of: governance, economic, social, environmental")
):
    """
    Process multiple documents using existing S3 object keys with client-specified document types and GRI type filtering.
    
    Each document type should be one of: 'sustainability_report', 'annual_report', 'financial_statement'
    If document_types is not provided, all documents will be treated as 'sustainability_report'
    If gri_type is provided, only indicators of that type will be evaluated
    """
    # Validate document type
    if request.document_types:
        for doc_type in request.document_types:
            if doc_type and doc_type not in ["sustainability_report", "annual_report", "financial_statement"]:
                raise HTTPException(status_code=400, detail=f"Invalid document type: {doc_type}")
    
    # Filter indices by gri_type if provided
    if gri_type:
        indices_to_process = [code for code, rule in scoring_rules.items() if rule["types"] == gri_type]
    else:
        indices_to_process = list(scoring_rules.keys())
    
    # Track overall processing time
    start_time = time.time()

    try:
        documents = []
        file_details = []
        
        # Track extraction time
        extraction_start = time.time()
        
        # Process each S3 object key
        for i, s3_object_key in enumerate(request.s3_object_keys):
            # Get PDF content from S3
            pdf_content = await get_pdf_from_s3(s3_object_key)
            
            if not pdf_content:
                raise HTTPException(status_code=404, detail=f"Failed to retrieve document from S3: {s3_object_key}")
            
            # Extract text from the document
            extracted_text = await extract_pdf_text(pdf_content)
            
            # Get extraction quality
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            extraction_quality = check_extraction_quality(extracted_text, pdf_reader)
            
            # Use client-provided document type or default to sustainability_report
            doc_type = "sustainability_report"  # Default type
            if request.document_types and i < len(request.document_types) and request.document_types[i]:
                # Validate the document type
                if request.document_types[i] in ["sustainability_report", "annual_report", "financial_statement"]:
                    doc_type = request.document_types[i]
            
            # Use provided filename or fallback to s3_object_key
            filename = s3_object_key
            if request.filenames and i < len(request.filenames) and request.filenames[i]:
                filename = request.filenames[i]
            
            # Store file details for database
            file_details.append({
                "filename": filename,
                "s3_object_key": s3_object_key,
                "file_size": len(pdf_content),
                "extraction_quality": extraction_quality,
                "document_type": doc_type
            })
            
            documents.append({
                "filename": filename,
                "text": extracted_text,
                "type": doc_type
            })
        
        extraction_time = time.time() - extraction_start
        
        # Track token usage across all evaluations
        total_tokens_used = 0
        token_usage_by_indicator = {}
        
        # Track AI processing times for each indicator
        ai_processing_times = {}
        ai_evaluation_start = time.time()
        
        # Build context for each indicator based on document types
        results = {}
        for indicator_code in indices_to_process:
            indicator = scoring_rules[indicator_code]
            try:
                await asyncio.sleep(1)  # Rate limiting for API calls
                
                # Track time for this specific indicator
                indicator_start = time.time()
                
                # Select appropriate document or documents for this indicator
                context = build_indicator_context(documents, indicator)
                
                # Evaluate the indicator with the selected context
                score, reasoning, token_count = await evaluate_indicator_with_context(context, indicator_code, indicator)
                
                # Calculate and store processing time
                indicator_time = time.time() - indicator_start
                ai_processing_times[indicator_code] = round(indicator_time, 2)
                
                # Store token usage information
                token_usage_by_indicator[indicator_code] = token_count
                if token_count["total_tokens"]:
                    total_tokens_used += token_count["total_tokens"]
                
                # Store results
                results[indicator_code] = {
                    "score": score,
                    "reasoning": reasoning,
                    "source_documents": context["source_documents"],
                    "title": indicator["disclosure"],
                    "type": indicator["types"],
                    "sub_type": indicator.get("sub-title", "Unknown"),
                    "description": indicator.get("description", ""),
                    "token_usage": token_count
                }
            except Exception as e:
                # Handle errors for individual indicators instead of failing the whole process
                error_message = str(e)
                if "429" in error_message:  # Rate limit error
                    # Wait longer if we hit a rate limit
                    await asyncio.sleep(5)
                    # Try again for this indicator
                    continue
                    
                # Log the error but continue with other indicators
                results[indicator_code] = {
                    "score": 0,
                    "title": indicator.get("disclosure", "Unknown"),
                    "type": indicator.get("types", "Unknown"),
                    "sub_type": indicator.get("sub-title", "Unknown"),
                    "description": indicator.get("description", ""),
                    "error": error_message
                }
        
        ai_evaluation_time = time.time() - ai_evaluation_start
        
        # Calculate summary scores by category
        summary = calculate_summary_scores(results)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Add timing metrics to response (matching /evaluate format)
        timing_metrics = {
            "total_processing_time_seconds": round(total_time, 2),
            "s3_upload_time_seconds": 0,  # Client handles uploads
            "extraction_time_seconds": round(extraction_time, 2),
            "extraction_quality_check_time_seconds": 0,  # Not tracked separately for multi
            "ai_evaluation_time_seconds": round(ai_evaluation_time, 2),
            "indicator_processing_times": ai_processing_times,
            "db_save_time_seconds": 0  # Will be updated below
        }

        # Prepare token usage data
        token_usage_data = {
            "total_tokens_used": total_tokens_used,
            "by_indicator": token_usage_by_indicator
        }

        # Save main document to database (using first file as primary document)
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
                performance_metrics=timing_metrics
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
            "performance_metrics": timing_metrics
        }
    except Exception as e:
        # Calculate time even for errors
        error_time = time.time() - start_time
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

def build_indicator_context(documents, indicator):
    """
    Build evaluation context for an indicator by selecting appropriate document(s).
    
    Args:
        documents: List of extracted documents with their types
        indicator: The indicator to evaluate
    
    Returns:
        dict: Context object with text to evaluate and metadata
    """
    indicator_type = indicator["types"]  # governance, environmental, etc.
    indicator_code = indicator.get("sub-title", "")
    
    # Initialize context
    context = {
        "combined_text": "",
        "source_documents": []
    }
    
    # Document type preference map for indicator types
    preference_map = {
        "governance": ["sustainability_report", "annual_report", "financial_statement"],
        "environmental": ["sustainability_report", "annual_report"],
        "social": ["sustainability_report", "annual_report"],
        "economic": ["financial_statement", "annual_report", "sustainability_report"]
    }
    
    # Get ordered preference for this indicator type
    preferences = preference_map.get(indicator_type, ["sustainability_report", "annual_report", "financial_statement"])
    
    # First try to find ideal document type for this indicator
    primary_docs = [doc for doc in documents if doc["type"] in preferences[:1]]
    
    # If no primary document found, use the preference order
    if not primary_docs:
        for pref in preferences:
            docs_of_type = [doc for doc in documents if doc["type"] == pref]
            if docs_of_type:
                primary_docs = docs_of_type
                break
    
    # If still no document found, use any available document
    if not primary_docs and documents:
        primary_docs = [documents[0]]
    
    # Build combined text from selected document(s)
    all_text = ""
    for doc in primary_docs:
        context["source_documents"].append(doc["filename"])
        all_text += doc["text"] + "\n\n"
    
    context["combined_text"] = all_text
    return context

async def evaluate_indicator_with_context(context, indicator_code, indicator):
    """
    Evaluate an indicator using context built from multiple documents.
    
    Args:
        context: Context object with text and metadata
        indicator_code: Code of the indicator to evaluate
        indicator: The indicator configuration
        
    Returns:
        tuple: (score, reasoning, token_count)
    """
    combined_text = context["combined_text"]
    
    # If no text is available, return a default score
    if not combined_text.strip():
        return 0, "No relevant text found in provided documents.", {"total_tokens": 0, "prompt_tokens": 0, "response_tokens": 0}
    
    # Find relevant sections based on keywords (similar to evaluate_indicator)
    keywords = indicator['keywords']
    relevant_sections = []
    text_lower = combined_text.lower()
    
    # Try to find sections containing keywords
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            # Find position of keyword
            index = text_lower.find(keyword_lower)
            # Extract surrounding context (2000 chars before and after)
            start = max(0, index - 2000)
            end = min(len(combined_text), index + 2000)
            relevant_sections.append(combined_text[start:end])
            
            # Look for more instances if document is large
            if len(combined_text) > 10000:
                # Find a second instance further in the document
                second_index = text_lower.find(keyword_lower, index + 100)
                if second_index > -1 and second_index != index:
                    start2 = max(0, second_index - 2000)
                    end2 = min(len(combined_text), second_index + 2000)
                    relevant_sections.append(combined_text[start2:end2])
    
    # If no relevant sections with keywords were found, use the beginning of the document
    if not relevant_sections:
        relevant_sections = [combined_text[:8000]]
    
    # Process and combine sections (same as in evaluate_indicator)
    unique_sections = []
    for section in relevant_sections[:4]:
        if section not in unique_sections:
            unique_sections.append(section)
    
    combined_text = "\n\n[...]\n\n".join(unique_sections)
    
    if len(combined_text) > 8000:
        combined_text = combined_text[:8000]
    
    # Use the existing evaluation logic from here (almost identical to evaluate_indicator)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Create evaluation prompt
    prompt = f"""
    You are an ESG (Environmental, Social, Governance) scoring expert. 
    Analyze the following extracted text against the indicator: {indicator_code} - {indicator['disclosure']}.
    
    Indicator description: {indicator['description']}
    
    Relevant keywords to look for: {', '.join(indicator['keywords'])}
    
    Scoring criteria:
    0: {indicator['criteria']['0']}
    1: {indicator['criteria'].get('1', 'Not specified')}
    2: {indicator['criteria'].get('2', 'Not specified')}
    3: {indicator['criteria'].get('3', 'Not specified')}
    4: {indicator['criteria']['4']}
    
    Based on the text below, assign a score from 0-4 and provide a brief explanation (2-3 sentences) justifying your score.
    
    First give your score as a single digit (0-4), then on a new line provide your explanation.
    
    Example:
    3
    The text clearly describes procedures for waste management including recycling programs. It provides quantitative data on waste reduction but lacks complete information on circular economy implementation.
    
    {combined_text}
    """
    
    # Get token count and make API request
    prompt_token_count = model.count_tokens(prompt).total_tokens
    
    try:
        response = await model.generate_content_async(prompt)
        
        # Parse response to extract score and reasoning
        response_text = response.text.strip()
        lines = response_text.split("\n", 1)
        
        # Extract score from first line
        score = 0
        for char in lines[0].strip():
            if char.isdigit() and int(char) in [0, 1, 2, 3, 4]:
                score = int(char)
                break
        
        # Extract reasoning from remaining text
        reasoning = lines[1].strip() if len(lines) > 1 else "No explanation provided."
        
        # Get token usage
        token_count = {
            "total_tokens": prompt_token_count + (len(response_text) // 4),
            "prompt_tokens": prompt_token_count,
            "response_tokens": len(response_text) // 4
        }
        
        return score, reasoning, token_count
        
    except Exception as e:
        return 0, f"Error: {str(e)}", {"total_tokens": prompt_token_count, "prompt_tokens": prompt_token_count, "response_tokens": 0}

def calculate_summary_scores(results):
    """Calculate summary scores by category"""
    # Calculate total score (SPDI index)
    total_score = sum(result["score"] for result in results.values())
    
    # Create a minimal summary with just the SPDI index
    summary = {
        "spdi_index": total_score
    }
    
    return summary

async def extract_pdf_text(pdf_content):
    """Extract text from PDF with fallback to Gemini for scanned documents"""
    # First try conventional extraction with PyPDF2
    pdf_file = io.BytesIO(pdf_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    # Extract text using PyPDF2
    extracted_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text() or ""
        extracted_text += page_text + "\n\n"
    
    # Check if this is likely a scanned document
    avg_chars_per_page = len(extracted_text) / len(pdf_reader.pages) if len(pdf_reader.pages) > 0 else 0
    
    # If it's a scanned document (low character count), use Gemini's image processing
    if avg_chars_per_page < 200:
        print(f"Detected potential scanned PDF (avg {avg_chars_per_page:.2f} chars per page). Using Gemini for image processing...")
        
        # Save PDF content to a temporary file (required for PyMuPDF)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_content)
            temp_pdf_path = temp_pdf.name
        
        try:
            # Extract images from PDF using PyMuPDF
            doc = fitz.open(temp_pdf_path)
            gemini_text = ""
            
            # Configure Gemini model for image processing
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Process each page as an image
            for page_num in range(len(doc)):
                # Convert PDF page to image
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality
                
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Convert image to base64 for Gemini
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                
                # Process with Gemini (with rate limiting)
                try:
                    await asyncio.sleep(1)  # Rate limiting
                    response = await model.generate_content_async([
                        "Extract all text visible in this image. Format as plain text with paragraphs preserved.",
                        {"mime_type": "image/png", "data": img_base64}
                    ])
                    gemini_text += response.text + "\n\n"
                except Exception as e:
                    print(f"Error processing page {page_num+1} with Gemini: {e}")
            
            # Clean up temporary file
            import os
            os.unlink(temp_pdf_path)
            
            # If Gemini extracted text, use it. Otherwise, fall back to PyPDF2 results
            if len(gemini_text.strip()) > len(extracted_text.strip()):
                return gemini_text
        except Exception as e:
            print(f"Error in image-based extraction: {e}")
    
    # Return the best text we have (either PyPDF2 or Gemini)
    return extracted_text

# Related to Database endpoints
@app.get("/documents")
async def list_documents(limit: int = 100, offset: int = 0):
    """Get a list of all analyzed documents with individual indicator scores and SPDI index"""
    try:
        with SessionLocal() as session:
            # Build base query for Documents
            query = session.query(Document).order_by(Document.id.desc())
            
            # Get total count BEFORE pagination
            total_count = query.count()
            
            # Apply pagination
            documents = query.offset(offset).limit(limit).all()
            
            result = []
            for doc in documents:
                # Get all indicator scores for this document
                indicators = {}
                for indicator in doc.analysis_results:
                    indicators[indicator.indicator_code] = {
                        "score": indicator.score,
                        "title": indicator.indicator_title,
                        "type": indicator.indicator_type,
                        "subtype": indicator.indicator_subtype,
                        "description": indicator.indicator_description,
                        "reasoning": indicator.reasoning
                    }
                
                # Get SPDI index score from score_summary
                spdi_index = doc.score_summary.spdi_index_score if doc.score_summary else 0
                
                # Create a document entry with detailed indicator scores and SPDI index
                doc_entry = {
                    "id": doc.id,
                    "filename": doc.filename,
                    "created_at": doc.upload_date.isoformat(),
                    "file_size": doc.file_size,
                    "spdi_index": spdi_index,
                    # All individual indicator scores
                    "indicators": indicators
                }
                result.append(doc_entry)
                
            return {
                "documents": result,
                "count": total_count  # Total documents in database, not just current page
            }
    except Exception as e:
        print(f"Error getting documents: {e}")
        return {"documents": [], "count": 0}

@app.get("/documents/{document_id}")
async def get_document(document_id: int):
    """Get the complete analysis results for a document"""
    try:
        document = get_document_analysis(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@app.get("/documents/{document_id}/pdf")
async def get_document_pdf(document_id: int):
    """Get a presigned URL to access the original PDF"""
    try:
        from aws import generate_presigned_url
        document = get_document_analysis(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
        
        url = await generate_presigned_url(document["s3_object_key"])
        
        if not url:
            raise HTTPException(status_code=500, detail="Failed to generate URL")
            
        return {"url": url, "expires_in": "1 hour"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating URL: {str(e)}")


# Utility endpoints
@app.get("/scoring-rules")
async def get_scoring_rules():
    """Return all scoring rules"""
    return scoring_rules

@app.get("/categories")
async def get_categories():
    """Return all available ESG categories"""
    categories = {}
    for indicator_code, indicator in scoring_rules.items():
        category = indicator["types"]
        sub_category = indicator["sub-title"]
        
        if category not in categories:
            categories[category] = {}
        
        if sub_category not in categories[category]:
            categories[category][sub_category] = []
        
        categories[category][sub_category].append({
            "code": indicator_code,
            "title": indicator["disclosure"]
        })
    
    return categories

class UpdateAnalysisResultRequest(BaseModel):
    score: Optional[int] = None
    reasoning: Optional[str] = None

@app.patch("/documents/{document_id}/indicator/{indicator_code}")
async def update_analysis_result(
    document_id: int,
    indicator_code: str,
    update: UpdateAnalysisResultRequest
):
    """
    Update an analysis result for a specific document and indicator code.
    """
    db = SessionLocal()
    try:
        # Find the analysis result
        ar = db.query(AnalysisResult).filter(
            AnalysisResult.document_id == document_id,
            AnalysisResult.indicator_code == indicator_code
        ).first()
        
        if not ar:
            raise HTTPException(status_code=404, detail=f"AnalysisResult not found for document {document_id} and indicator {indicator_code}")
        
        # Update the fields that were provided
        update_data = update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(ar, field, value)
        
        # Validate score if provided
        if 'score' in update_data and (update_data['score'] < 0 or update_data['score'] > 4):
            raise HTTPException(status_code=400, detail="Score must be between 0 and 4")
        
        db.commit()
        
        # Recalculate the SPDI index for the document
        total_spdi_index = sum(r.score for r in db.query(AnalysisResult).filter(AnalysisResult.document_id == document_id))
        
        # Update the score summary
        if ar.document.score_summary:
            ar.document.score_summary.spdi_index_score = total_spdi_index
        else:
            # Create score summary if it doesn't exist
            from neon import ScoreSummary
            score_summary = ScoreSummary(
                document_id=document_id,
                spdi_index_score=total_spdi_index
            )
            db.add(score_summary)
        
        db.commit()
        
        return {
            "success": True,
            "message": f"AnalysisResult updated successfully for indicator {indicator_code}",
            "updated_spdi_index": total_spdi_index
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating analysis result: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)