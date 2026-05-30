import pytest
from unittest.mock import patch, AsyncMock


@patch("routes.pdf.upload_to_s3", new_callable=AsyncMock)
def test_upload_pdf_success(mock_upload, client):
    """Test legacy server-side PDF upload returns generated object key."""
    mock_upload.return_value = "new-s3-key"
    
    file_payload = {"pdf": ("test.pdf", b"pdf dummy data", "application/pdf")}
    response = client.post("/upload", files=file_payload)
    assert response.status_code == 200
    assert response.json()["s3_object_key"] == "new-s3-key"
    mock_upload.assert_called_once_with(b"pdf dummy data", "test.pdf")


@patch("routes.pdf.upload_to_s3", new_callable=AsyncMock)
def test_upload_pdf_failure(mock_upload, client):
    """Test legacy upload returns 500 when S3 client upload fails."""
    mock_upload.return_value = None
    
    file_payload = {"pdf": ("test.pdf", b"pdf dummy data", "application/pdf")}
    response = client.post("/upload", files=file_payload)
    assert response.status_code == 500


@patch("routes.pdf.generate_presigned_upload_url", new_callable=AsyncMock)
def test_get_upload_presigned_url_success(mock_presign, client):
    """Test obtaining a presigned client PUT URL successfully."""
    mock_presign.return_value = "https://s3.example.com/presigned-put-url"
    
    response = client.get("/upload/presign?filename=report.pdf")
    assert response.status_code == 200
    data = response.json()
    assert data["presigned_url"] == "https://s3.example.com/presigned-put-url"
    assert "object_key" in data
    assert data["filename"] == "report.pdf"


@patch("routes.pdf.generate_presigned_upload_url", new_callable=AsyncMock)
def test_get_upload_presigned_url_failure(mock_presign, client):
    """Test presign requests return 500 on S3 connection failures."""
    mock_presign.return_value = None
    
    response = client.get("/upload/presign?filename=report.pdf")
    assert response.status_code == 500


@patch("routes.pdf.get_pdf_from_s3", new_callable=AsyncMock)
def test_get_pdf_stream_success(mock_get_pdf, client):
    """Test streaming raw PDF bytes from an S3 key."""
    mock_get_pdf.return_value = b"raw pdf bytes"
    
    response = client.get("/pdf/some-s3-key.pdf")
    assert response.status_code == 200
    assert response.content == b"raw pdf bytes"
    assert response.headers["content-type"] == "application/pdf"
    mock_get_pdf.assert_called_once_with("some-s3-key.pdf")


@patch("routes.pdf.get_pdf_from_s3", new_callable=AsyncMock)
def test_get_pdf_stream_not_found(mock_get_pdf, client):
    """Test streaming returns 404 when key doesn't exist in S3."""
    mock_get_pdf.return_value = None
    
    response = client.get("/pdf/missing.pdf")
    assert response.status_code == 404
