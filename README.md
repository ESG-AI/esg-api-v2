# ESG Scoring Backend v2  

A powerful backend service for automated Environmental, Social, and Governance (ESG) scoring of corporate sustainability reports using AI analysis.  

## Overview  
This project provides an API for evaluating sustainability reports and other corporate documents against standardized ESG indicators based on the Global Reporting Initiative (GRI) framework. It leverages Google's Gemini AI models to analyze documents and provide quantitative scores with qualitative reasoning.  

## Key Features  
- **Automated ESG Scoring**: Evaluate documents against 40+ GRI indicators  
- **Multi-document Analysis**: Process sustainability reports, annual reports, and financial statements together  
- **Document Type Intelligence**: Selects the appropriate document for each indicator based on content type  
- **OCR Capability**: Extracts text from scanned PDFs using Gemini's image processing  
- **Reference-based Scoring**: Uses real-world examples for consistent scoring benchmarks  
- **Detailed Reasoning**: Provides explanations for each score to ensure transparency  
- **Document Storage**: Saves uploaded documents to S3 for future reference  
- **Database Integration**: Stores all analysis results in a PostgreSQL database  
- **Performance Metrics**: Tracks processing time and token usage for optimization  

## Installation  

### Prerequisites  
- Python 3.10+  
- PostgreSQL database (or Neon.tech account)  
- AWS S3 bucket  
- Google AI API key (for Gemini)  

### Setup  
1. Clone the repository:  
    ```bash  
    git clone <repository-url>  
    cd esg-scoring-backend-v2  
    ```  

2. Create and activate a virtual environment:  
    ```bash  
    python -m venv venv  
    source venv/bin/activate  # On Windows: venv\Scripts\activate  
    ```  

3. Install dependencies:  
    ```bash  
    pip install -r requirements.txt  
    ```  

4. Set up environment variables:  
    Create a `.env` file with the required configurations (e.g., database URL, S3 credentials, API keys).  

5. Initialize the database:  
    ```bash  
    alembic upgrade head  
    ```  

## Usage  

### Running the Server  
Start the development server:  
```bash  
uvicorn app.main:app --reload  
```  
The API will be available at `http://localhost:8000`.  

### API Endpoints  

#### Document Analysis  
- `POST /extract`: Extract text from a PDF with quality diagnostics  
- `POST /evaluate`: Evaluate a single PDF against all ESG indicators  
- `POST /evaluate-multi`: Process multiple documents with specified document types  

#### Document Management  
- `GET /documents`: List all analyzed documents  
- `GET /documents/{document_id}`: Get complete analysis results for a document  
- `GET /documents/{document_id}/pdf`: Get a presigned URL to access the original PDF  

#### Utility Endpoints  
- `GET /scoring-rules`: Return all scoring rules  
- `GET /categories`: Return all available ESG categories  

### Example: Evaluating a Document  
```bash  
curl -X POST http://localhost:8000/evaluate -F "file=@sustainability_report.pdf"  
```  

### Example: Multi-document Evaluation  
```bash  
curl -X POST http://localhost:8000/evaluate-multi -F "files=@report1.pdf,@report2.pdf"  
```  

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
- **Google Gemini AI**: For document analysis and scoring  
- **PyPDF2 & PyMuPDF**: PDF text extraction  
- **SQLAlchemy**: Database ORM  
- **Neon PostgreSQL**: Database storage  
- **AWS S3**: Document storage  

### Available Gemini Models  
The system uses different Gemini models for different tasks:  
- `gemini-2.5-pro`: For core ESG analysis and OCR processing  
- `gemini-1.5-flash-8b`: For simpler text extraction tasks  

## License  
This project is licensed under the [MIT License](LICENSE).  

## Contributing  
Contributions are welcome! Please feel free to submit a Pull Request.  