from fastapi import FastAPI, UploadFile, File, HTTPException
import google.generativeai as genai
import json
import os
import io
import PyPDF2
import asyncio
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

print(f"Gemini SDK version: {genai.__version__}")

# Load scoring rules from JSON file
with open("scoring_rules.json", "r") as f:
    scoring_rules = json.load(f)

@app.post("/extract")
async def extract_pdf(pdf: UploadFile = File(...)):
    """
    Extract text from a PDF and return detailed extraction diagnostics.
    This endpoint is for testing extraction quality without running the full ESG evaluation.
    """
    try:
        pdf_content = await pdf.read()
        
        # Extract text from PDF using PyPDF2
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        extracted_text = ""
        page_details = []
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text() or ""
            extracted_text += page_text + "\n\n"
            
            # Track per-page stats
            page_details.append({
                "page_number": page_num + 1,
                "characters": len(page_text),
                "words": len(page_text.split()) if page_text else 0,
                "empty": len(page_text.strip()) == 0
            })
        
        # Check extraction quality
        extraction_quality = check_extraction_quality(extracted_text, pdf_reader)
        
        # Get sample text (first 1000 chars)
        text_sample = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
        
        return {
            "filename": pdf.filename,
            "extraction_quality": extraction_quality,
            "page_details": page_details,
            "text_sample": text_sample,
            "text_length": len(extracted_text),
            "page_count": len(pdf_reader.pages),
            "empty_pages": sum(1 for page in page_details if page["empty"])
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

@app.post("/evaluate")
async def evaluate_pdf(pdf: UploadFile = File(...)):
    """
    Evaluate a single PDF sustainability report against all ESG indices.
    Makes individual Gemini API requests for each index to ensure focused evaluation.
    Includes token usage information for API monitoring.
    """
    try:
        pdf_content = await pdf.read()
        
        # Extract text from PDF using PyPDF2 instead of Gemini API
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        extracted_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:  # Some pages might return None
                extracted_text += page_text + "\n\n"
                
        # Check extraction quality
        extraction_quality = check_extraction_quality(extracted_text, pdf_reader)
        
        if not extraction_quality["extraction_success"]:
            # You could either return a warning or raise an error depending on severity
            # Here we'll include a warning in the response but continue processing
            extraction_warning = f"Warning: PDF extraction issues detected: {', '.join(extraction_quality['extraction_issues'])}"
        else:
            extraction_warning = None
        
        # Rest of your existing code remains the same...
        # Track token usage across all evaluations
        total_tokens_used = 0
        token_usage_by_indicator = {}
        
        # Evaluate against all indicators with individual API calls for each index
        results = {}
        for indicator_code, indicator in scoring_rules.items():
            try:
                await asyncio.sleep(1)  # Rate limiting for API calls
                score, reasoning, token_count = await evaluate_indicator(extracted_text, indicator_code, indicator)
                
                # Store token usage information
                token_usage_by_indicator[indicator_code] = token_count
                if token_count["total_tokens"]:
                    total_tokens_used += token_count["total_tokens"]
                
                results[indicator_code] = {
                    "score": score,
                    "reasoning": reasoning,  # Include reasoning in results
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
        
        # Calculate summary scores by category
        summary = calculate_summary_scores(results)
        
        return {
            "filename": pdf.filename,
            "extraction_quality": extraction_quality,
            "extraction_warning": extraction_warning,
            "indicators": results,
            "summary": summary,
            "token_usage": {
                "total_tokens_used": total_tokens_used,
                "by_indicator": token_usage_by_indicator
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

async def evaluate_indicator(text, indicator_code, indicator):
    """
    Evaluate a specific ESG indicator using Gemini AI with keyword-based context extraction.
    Finds the most relevant parts of the document by locating ESG index keywords.
    Returns both score and token usage information.
    """
    
    # Configure Gemini model for this specific evaluation
    model = genai.GenerativeModel('gemini-1.5-flash-8b')
    
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
    
    Based on the text below, assign a score from 0-4 and provide a brief explanation (2-3 sentences) justifying your score.
    
    First give your score as a single digit (0-4), then on a new line provide your explanation.
    
    Example:
    3
    The text clearly describes procedures for waste management including recycling programs. It provides quantitative data on waste reduction but lacks complete information on circular economy implementation.
    
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
        
        return score, reasoning, token_count
        
    except Exception as e:
        return 0, f"Error: {str(e)}", {"total_tokens": prompt_token_count, "prompt_tokens": prompt_token_count, "response_tokens": 0}

def calculate_summary_scores(results):
    """Calculate summary scores by category"""
    categories = {
        "governance": {"total": 0, "count": 0},
        "economic": {"total": 0, "count": 0},
        "social": {"total": 0, "count": 0},
        "environmental": {"total": 0, "count": 0}
    }
    
    for indicator_code, result in results.items():
        category = result["type"]
        if category in categories:
            categories[category]["total"] += result["score"]
            categories[category]["count"] += 1
    
    summary = {}
    for category, data in categories.items():
        if data["count"] > 0:
            summary[category] = round(data["total"] / data["count"], 2)
        else:
            summary[category] = 0
    
    # Calculate overall score
    total_scores = sum(data["total"] for data in categories.values())
    total_indicators = sum(data["count"] for data in categories.values())
    summary["overall"] = round(total_scores / total_indicators, 2) if total_indicators > 0 else 0
    
    return summary

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)