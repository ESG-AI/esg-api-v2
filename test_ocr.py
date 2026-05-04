import asyncio
import io
import os
from main import extract_pdf_text
from PIL import Image, ImageDraw, ImageFont
import tempfile
import fitz

# 1. Create a dummy image with some text
img = Image.new('RGB', (400, 200), color=(255, 255, 255))
d = ImageDraw.Draw(img)
d.text((10,10), "This is a test of the ESG OCR.", fill=(0,0,0))
d.text((10,50), "If this works, Gemini is successfully extracting text.", fill=(0,0,0))
d.text((10,90), "We need to ensure it handles scanned PDFs correctly.", fill=(0,0,0))

# 2. Save image as PDF (this makes it a scanned PDF with no extractable text by PyPDF2)
pdf_bytes = io.BytesIO()
img.save(pdf_bytes, "PDF", resolution=100.0)
pdf_content = pdf_bytes.getvalue()

# 3. Test extraction
async def run_test():
    print("Starting extraction...")
    result = await extract_pdf_text(pdf_content)
    print("--- EXTRACTED TEXT ---")
    print(result)
    print("----------------------")

if __name__ == "__main__":
    asyncio.run(run_test())
