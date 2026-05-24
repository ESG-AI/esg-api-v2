# ESG Scoring Backend v2

A powerful backend service for automated Environmental, Social, and Governance (ESG) scoring of corporate sustainability reports using AI analysis.

## Overview

This project provides an API for evaluating sustainability reports and other corporate documents against standardized ESG indicators based on the Global Reporting Initiative (GRI) framework. It leverages OpenAI and Google's Gemini AI models to analyze documents and provide quantitative scores with qualitative reasoning.

## Key Features

- **Automated ESG Scoring**: Evaluate documents against 40+ GRI indicators
- **Multi-document Analysis**: Process sustainability reports, annual reports, and financial statements together
- **Document Type Intelligence**: Selects the appropriate document for each indicator based on content type
- **OCR Capability**: Extracts text from scanned PDFs using OpenAI or Gemini's image processing capabilities
- **Reference-based Scoring**: Uses real-world examples for consistent scoring benchmarks
- **Detailed Reasoning**: Provides explanations for each score to ensure transparency
- **Document Storage**: Saves uploaded documents to S3 for future reference
- **Database Integration**: Stores all analysis results in a PostgreSQL database
- **Performance Metrics**: Tracks processing time and token usage for optimization

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (fast Python package and project manager)
- PostgreSQL database (or Neon.tech account)
- AWS S3 bucket
- OpenAI API key
- Google AI API key (if using Gemini models)

### Setup

1. Install `uv` if you haven't already:

   - **macOS/Linux:**
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
     *(Or via Homebrew: `brew install uv`)*

   - **Windows:**
     ```powershell
     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```

2. Clone the repository:

   ```bash
   git clone https://github.com/ESG-AI/esg-api-v2.git
   cd esg-api-v2
   ```

3. Install dependencies and set up the virtual environment:

   ```bash
   uv sync
   ```

4. Set up environment variables:  
   Create a `.env` file with the required configurations (e.g., database URL, S3 credentials, API keys).

## Usage

### Running the Server

Start the development server using `uv run`:

```bash
uv run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### API Documentation

The API provides three different ways to access the documentation:

1. **Interactive Swagger UI** (Recommended for testing)
   - Visit `http://localhost:8000/docs`
   - Features:
     - Interactive API testing
     - Request/response schemas
     - Authentication support
     - Try-it-out functionality
     - Example requests

2. **ReDoc** (Recommended for reading)
   - Visit `http://localhost:8000/redoc`
   - Features:
     - Clean, responsive interface
     - Detailed endpoint descriptions
     - Request/response examples
     - Authentication requirements
     - Better for documentation reading

3. **OpenAPI Specification**
   - Visit `http://localhost:8000/openapi.json`
   - Features:
     - Raw OpenAPI 3.0 specification
     - Machine-readable format
     - Useful for API clients
     - Can be imported into API tools

#### Authentication

All API endpoints require authentication using an API key. Include the key in your requests:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/your-endpoint
```

#### Example: Using the Documentation

1. Start the server: `uv run uvicorn main:app --reload`
2. Open `http://localhost:8000/docs` in your browser
3. Click on any endpoint to expand its details
4. Click "Try it out" to test the endpoint
5. Fill in the required parameters
6. Click "Execute" to make the request
7. View the response and status code

### Batch Processing (Local)

For testing and local evaluation, a standalone batch script (`batch_process.py`) is provided. This script evaluates a folder of PDFs simultaneously using OpenAI's gpt-4o-mini model and outputs the individual indicator scores to a CSV file.

To run batch scoring locally:

1. Ensure you have a directory named `test_pdfs` in the root of the project.
2. Put your input PDF files into the `test_pdfs` folder.
3. Run the batch script command:
   ```bash
   uv run python batch_process.py
   ```
4. Once completed, the result will be saved as a `batch_results.csv` file in the project root containing the evaluation scores for each document.

## Scoring System

The scoring system is based on the GRI standards framework and uses a 0-4 scale:

- **0**: No information provided
- **1**: Minimal information (25% of requirements)
- **2**: Partial information (50% of requirements)
- **3**: Substantial information (75% of requirements)
- **4**: Complete information (100% of requirements)

Each indicator has specific scoring criteria defined in the `scoring_rules.json` file, along with reference examples for consistent evaluation.

## Technical Architecture

- **FastAPI**: Web framework for API endpoints
- **OpenAI & Google Gemini**: For document analysis and scoring
- **PyPDF2 & PyMuPDF**: PDF text extraction
- **SQLAlchemy**: Database ORM
- **Neon PostgreSQL**: Database storage
- **AWS S3**: Document storage

### Available AI Models

The system supports multiple AI models for evaluation:

**OpenAI (Default):**

- `gpt-4o-mini` (used for all evaluation and batch processing)
- `gpt-4o`

**Google Gemini:**

- `gemini-1.5-pro`

These models are used for:

- ESG analysis and scoring
- OCR processing of scanned documents
- Text extraction and analysis
