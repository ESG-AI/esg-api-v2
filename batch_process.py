import os
import asyncio
import io
import csv
import time
import PyPDF2
from pathlib import Path

# Important: We must load env vars before importing from main since main initializes the clients
from dotenv import load_dotenv
load_dotenv()

# Import the core logic from the FastAPI app directly
from main import (
    extract_pdf_text,
    check_extraction_quality,
    evaluate_indicator,
    scoring_rules,
    calculate_summary_scores
)

# Configuration
TEST_PDF_DIR = "./test_pdfs"
OUTPUT_CSV_FILE = "batch_results.csv"
MODEL_NAME = "gpt-4o-mini"  # Default model for large batch testing
CONCURRENCY_LIMIT = 5 # Number of simultaneous PDF files to process

async def process_single_pdf(file_path: Path):
    """Processes a single PDF file and returns the evaluation results."""
    print(f"Starting processing: {file_path.name}")
    start_time = time.time()
    
    try:
        # 1. Read the PDF from local disk
        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()
            
        # 2. Extract Text
        extracted_text = await extract_pdf_text(pdf_bytes)
        
        # 3. Quality Check (Optional but good for tracking)
        pdf_file_obj = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        quality = check_extraction_quality(extracted_text, pdf_reader)
        
        if not quality["extraction_success"]:
            print(f"Warning: Poor extraction quality for {file_path.name}")
            
        # 4. Evaluate all configured indicators
        results = {}
        total_tokens = 0
        ai_time = 0
        
        # Rate limit evaluation calls per document slightly to avoid overwhelming OpenAI completely
        for indicator_code, indicator_config in scoring_rules.items():
            indicator_start = time.time()
            score, reasoning, token_usage = await evaluate_indicator(
                text=extracted_text,
                indicator_code=indicator_code,
                indicator=indicator_config,
                model_name=MODEL_NAME
            )
            
            ai_time += (time.time() - indicator_start)
            total_tokens += token_usage.get("total_tokens", 0)
            
            results[indicator_code] = {
                "score": score,
                "reasoning": reasoning
            }
            # Optional small delay between indicators for a single file to keep rate limit safe
            await asyncio.sleep(0.5) 
            
        # Calculate summary metrics
        summary = calculate_summary_scores(results)
        total_time = time.time() - start_time
        
        print(f"Finished processing: {file_path.name} in {total_time:.2f}s (SPDI Index: {summary.get('spdi_index', 0)})")
        
        return {
            "filename": file_path.name,
            "success": True,
            "total_time_seconds": round(total_time, 2),
            "total_tokens_used": total_tokens,
            "spdi_index": summary.get("spdi_index", 0),
            "indicators": results
        }
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return {
            "filename": file_path.name,
            "success": False,
            "error": str(e)
        }

async def process_batch_with_semaphore(semaphore, file_path):
    """Wrapper to limit concurrent execution of PDF processing."""
    async with semaphore:
        return await process_single_pdf(file_path)

async def main():
    pdf_dir = Path(TEST_PDF_DIR)
    
    # Ensure directory exists
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"Error: Directory '{TEST_PDF_DIR}' does not exist.")
        print(f"Please create it and add your test PDFs before running the script.")
        return

    # Find all PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{TEST_PDF_DIR}'.")
        return
        
    print(f"Found {len(pdf_files)} PDF(s) to process. Starting batch evaluation using {MODEL_NAME}...\n")
    
    # Run the batch with a concurrency limit
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [process_batch_with_semaphore(semaphore, pdf_file) for pdf_file in pdf_files]
    
    # Gather responses
    all_results = await asyncio.gather(*tasks)
    
    # Filter out total failures if we just want to output successful scores
    successful_results = [r for r in all_results if r.get("success")]
    failed_results = [r for r in all_results if not r.get("success")]
    
    print(f"\nBatch processing complete. Success: {len(successful_results)}, Failed: {len(failed_results)}")
    
    if not successful_results:
        print("No successful results to export.")
        return
        
    print(f"Exporting results to {OUTPUT_CSV_FILE}...")
    export_to_csv(successful_results)
    print("Export complete!")

def export_to_csv(results):
    """Writes the aggregated results to a specific CSV format."""
    
    # Dynamically grab all indicator codes from the first successful result
    # to act as our dynamic headers.
    sample_indicators = results[0]["indicators"].keys()
    sorted_indicator_codes = sorted(list(sample_indicators))
    
    # Define CSV Headers
    headers = [
        "Filename", 
        "SPDI Index Score", 
        "Total Time (s)", 
        "Total Tokens"
    ]
    
    # Add columns for each individual score
    for code in sorted_indicator_codes:
        headers.append(f"Score: {code}")
        
    # Optional: Add reasoning columns at the end if you want human comparison notes
    # for code in sorted_indicator_codes:
    #     headers.append(f"Reasoning: {code}")
        
    with open(OUTPUT_CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        for result in results:
            row = [
                result["filename"],
                result["spdi_index"],
                result["total_time_seconds"],
                result["total_tokens_used"]
            ]
            
            # Map score logic
            for code in sorted_indicator_codes:
                indicator_data = result["indicators"].get(code, {})
                row.append(indicator_data.get("score", "N/A"))
                
            # Map reasoning logic
            # for code in sorted_indicator_codes:
            #     indicator_data = result["indicators"].get(code, {})
            #     row.append(indicator_data.get("reasoning", "N/A"))
                
            writer.writerow(row)

if __name__ == "__main__":
    # Standard python async entry point
    asyncio.run(main())
