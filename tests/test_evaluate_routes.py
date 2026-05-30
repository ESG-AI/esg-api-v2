import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from auth.security import create_access_token
from db.models import UserRole
from db.repositories.user import create_user, update_user_role


def create_test_user_helper(email, password, role=UserRole.free):
    from auth.security import hash_password
    user = create_user(email=email, hashed_password=hash_password(password))
    if role != UserRole.free:
        user = update_user_role(user.id, role)
    return user


@patch("routes.evaluate.PyPDF2.PdfReader")
@patch("routes.evaluate.extract_pdf_text", new_callable=AsyncMock)
def test_extract_pdf_route_success(mock_extract, mock_reader_cls, client):
    """Test extracting text from PDF route returns diagnostics."""
    # Make mock return values longer than 200 characters to avoid triggering Gemini OCR fallback
    long_text = "extracted content text representing a corporate ESG report. " * 5
    mock_extract.return_value = long_text
    
    # Setup mock PDF reader
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = long_text
    mock_reader.pages = [mock_page]
    mock_reader_cls.return_value = mock_reader
    
    user = create_test_user_helper("user@example.com", "pass123", UserRole.free)
    token = create_access_token(user_id=user.id, email=user.email, role=user.role.value)
    headers = {"Authorization": f"Bearer {token}"}
    
    file_payload = {"pdf": ("dummy.pdf", b"%PDF-1.4 dummy contents", "application/pdf")}
    response = client.post("/extract", files=file_payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "dummy.pdf"
    assert data["text_length"] > 0
    assert data["page_count"] == 1
    assert data["used_gemini_ocr"] is False


def test_evaluate_pdf_route_forbidden_for_free(client):
    """Test that free tier users are forbidden (403) from running evaluations."""
    user = create_test_user_helper("freeuser@example.com", "pass123", UserRole.free)
    token = create_access_token(user_id=user.id, email=user.email, role=user.role.value)
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.post(
        "/evaluate",
        headers=headers,
        json={"s3_object_key": "some-key", "filename": "report.pdf"}
    )
    assert response.status_code == 403


@patch("routes.evaluate.evaluate_all_indicators", new_callable=AsyncMock)
@patch("routes.evaluate.extract_pdf_text", new_callable=AsyncMock)
@patch("routes.evaluate.get_pdf_from_s3", new_callable=AsyncMock)
def test_evaluate_pdf_route_success_for_paid(mock_get_s3, mock_extract, mock_eval, client):
    """Test running a synchronous ESG evaluation as a paid tier user."""
    mock_get_s3.return_value = b"pdf binary content"
    mock_extract.return_value = "extracted report content"
    mock_eval.return_value = {
        "GRI-302-1": {"score": 4, "reasoning": "Excellent energy disclosure."}
    }
    
    user = create_test_user_helper("paiduser@example.com", "pass123", UserRole.paid)
    token = create_access_token(user_id=user.id, email=user.email, role=user.role.value)
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.post(
        "/evaluate",
        headers=headers,
        json={"s3_object_key": "some-key", "filename": "report.pdf"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_score"] == 4
    assert "results" in data
    assert data["results"]["GRI-302-1"]["score"] == 4
    
    mock_get_s3.assert_called_once_with("some-key")
    mock_extract.assert_called_once_with(b"pdf binary content")
    mock_eval.assert_called_once()
