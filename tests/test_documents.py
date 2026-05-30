import pytest
from unittest.mock import patch, AsyncMock
from auth.security import create_access_token
from db.models import UserRole
from db.repositories.user import create_user, update_user_role
from db.repositories.document import save_analysis_results, get_document_by_id


def create_test_user_helper(email, password, role=UserRole.free):
    from auth.security import hash_password
    user = create_user(email=email, hashed_password=hash_password(password))
    if role != UserRole.free:
        user = update_user_role(user.id, role)
    return user


def seed_document(user_id=None, key="some-key"):
    return save_analysis_results(
        filename="test.pdf",
        s3_object_key=key,
        file_size=1024,
        extraction_quality={"quality": "high"},
        results={"GRI-302-1": {"score": 3, "reasoning": "Energy details"}},
        summary={"total_score": 3},
        token_usage={"total_tokens": 100},
        performance_metrics={"time_taken": 2.5},
        user_id=str(user_id) if user_id else None
    )


def test_list_documents_normal_user(client):
    """Test that a regular user only sees their own documents in the list."""
    user1 = create_test_user_helper("user1@example.com", "pass123", UserRole.free)
    user2 = create_test_user_helper("user2@example.com", "pass123", UserRole.free)
    
    # Seed one doc for user1, one for user2
    doc_id1 = seed_document(user1.id, "key-1")
    seed_document(user2.id, "key-2")
    
    token = create_access_token(user_id=user1.id, email=user1.email, role=user1.role.value)
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.get("/documents", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["documents"][0]["id"] == doc_id1


def test_list_documents_admin(client):
    """Test that an admin user sees all user documents in the list."""
    user = create_test_user_helper("user@example.com", "pass123", UserRole.free)
    admin = create_test_user_helper("admin@example.com", "pass123", UserRole.admin)
    
    seed_document(user.id, "key-user")
    seed_document(admin.id, "key-admin")
    
    token = create_access_token(user_id=admin.id, email=admin.email, role=admin.role.value)
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.get("/documents", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 2


def test_get_document_success(client):
    """Test retrieving a single document analysis successfully."""
    user = create_test_user_helper("user@example.com", "pass123", UserRole.free)
    doc_id = seed_document(user.id)
    
    token = create_access_token(user_id=user.id, email=user.email, role=user.role.value)
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.get(f"/documents/{doc_id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["id"] == doc_id


def test_get_document_not_found(client):
    """Test retrieving a non-existent document analysis yields a 404 error."""
    user = create_test_user_helper("user@example.com", "pass123", UserRole.free)
    token = create_access_token(user_id=user.id, email=user.email, role=user.role.value)
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.get("/documents/9999", headers=headers)
    assert response.status_code == 404


@patch("routes.documents.generate_presigned_url", new_callable=AsyncMock)
def test_get_document_pdf_success(mock_presign, client):
    """Test generating a presigned PDF URL for an existing document."""
    mock_presign.return_value = "https://s3.example.com/test-presigned-url"
    
    user = create_test_user_helper("user@example.com", "pass123", UserRole.free)
    doc_id = seed_document(user.id)
    
    token = create_access_token(user_id=user.id, email=user.email, role=user.role.value)
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.get(f"/documents/{doc_id}/pdf", headers=headers)
    assert response.status_code == 200
    assert response.json()["url"] == "https://s3.example.com/test-presigned-url"
    mock_presign.assert_called_once_with("some-key")


@patch("routes.documents.generate_presigned_url", new_callable=AsyncMock)
def test_get_document_pdf_not_found(mock_presign, client):
    """Test generating a presigned PDF URL for a non-existent document yields 404."""
    user = create_test_user_helper("user@example.com", "pass123", UserRole.free)
    token = create_access_token(user_id=user.id, email=user.email, role=user.role.value)
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.get("/documents/9999/pdf", headers=headers)
    assert response.status_code == 404


def test_update_analysis_result_forbidden(client):
    """Test that a non-admin is forbidden (403) from updating indicator scores."""
    user = create_test_user_helper("user@example.com", "pass123", UserRole.free)
    doc_id = seed_document(user.id)
    
    token = create_access_token(user_id=user.id, email=user.email, role=user.role.value)
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.patch(
        f"/documents/{doc_id}/indicator/GRI-302-1",
        headers=headers,
        json={"score": 4, "reasoning": "Updated energy score"}
    )
    assert response.status_code == 403


def test_update_analysis_result_success(client):
    """Test that an admin can update indicator results on a document."""
    admin = create_test_user_helper("admin@example.com", "pass123", UserRole.admin)
    doc_id = seed_document(admin.id)
    
    token = create_access_token(user_id=admin.id, email=admin.email, role=admin.role.value)
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.patch(
        f"/documents/{doc_id}/indicator/GRI-302-1",
        headers=headers,
        json={"score": 4, "reasoning": "Updated energy score"}
    )
    assert response.status_code == 200
    
    # Verify in DB
    doc = get_document_by_id(doc_id)
    assert doc["indicators"]["GRI-302-1"]["score"] == 4
    assert doc["indicators"]["GRI-302-1"]["reasoning"] == "Updated energy score"

