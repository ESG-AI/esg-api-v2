import pytest
from unittest.mock import patch, MagicMock
from botocore.exceptions import ClientError
from infrastructure.s3 import upload_to_s3, download_from_s3, get_pdf_from_s3


@pytest.mark.asyncio
@patch("infrastructure.s3.s3_client")
async def test_upload_to_s3_success(mock_s3):
    # Mock put_object to simulate a successful S3 upload
    mock_s3.put_object.return_value = {}
    
    key = await upload_to_s3(b"test pdf content", "test.pdf")
    assert key is not None
    assert isinstance(key, str)
    mock_s3.put_object.assert_called_once()


@pytest.mark.asyncio
@patch("infrastructure.s3.s3_client")
async def test_upload_to_s3_failure(mock_s3):
    # Mock put_object to raise a ClientError to test error handling
    error_response = {"Error": {"Code": "500", "Message": "Internal Error"}}
    mock_s3.put_object.side_effect = ClientError(error_response, "PutObject")
    
    key = await upload_to_s3(b"test pdf content", "test.pdf")
    assert key is None


@pytest.mark.asyncio
@patch("infrastructure.s3.s3_client")
async def test_download_from_s3_success(mock_s3):
    # Mock get_object to return a mocked stream representing downloaded file bytes
    mock_body = MagicMock()
    mock_body.read.return_value = b"file bytes"
    mock_s3.get_object.return_value = {"Body": mock_body}
    
    content = await download_from_s3("some-key")
    assert content == b"file bytes"
    mock_s3.get_object.assert_called_once()


@pytest.mark.asyncio
@patch("infrastructure.s3.s3_client")
async def test_download_from_s3_failure(mock_s3):
    # Mock get_object to simulate NoSuchKey error
    error_response = {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}}
    mock_s3.get_object.side_effect = ClientError(error_response, "GetObject")
    
    content = await download_from_s3("missing-key")
    assert content is None


@pytest.mark.asyncio
@patch("infrastructure.s3.s3_client")
async def test_get_pdf_from_s3_success(mock_s3):
    # Mock get_object to return pdf file bytes
    mock_body = MagicMock()
    mock_body.read.return_value = b"pdf content"
    mock_s3.get_object.return_value = {"Body": mock_body}
    
    content = await get_pdf_from_s3("pdf-key")
    assert content == b"pdf content"


@pytest.mark.asyncio
@patch("infrastructure.s3.s3_client")
async def test_get_pdf_from_s3_failure_raises(mock_s3):
    # Mock get_object to raise a ClientError to test hard failure raising in the evaluation pipeline
    error_response = {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}
    mock_s3.get_object.side_effect = ClientError(error_response, "GetObject")
    
    with pytest.raises(RuntimeError) as exc_info:
        await get_pdf_from_s3("restricted-key")
    assert "Failed to download" in str(exc_info.value)
