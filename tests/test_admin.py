import pytest
from auth.security import create_access_token
from db.models import UserRole
from db.repositories.user import create_user, update_user_role, get_user_by_id


def create_test_user_helper(email, password, role=UserRole.free):
    from auth.security import hash_password
    user = create_user(email=email, hashed_password=hash_password(password))
    if role != UserRole.free:
        user = update_user_role(user.id, role)
    return user


def test_admin_list_users_forbidden(client):
    """Test listing users as a non-admin yields a 403 Forbidden response."""
    user = create_test_user_helper("normal@example.com", "pass123", UserRole.free)
    token = create_access_token(user_id=user.id, email=user.email, role=user.role.value)
    
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/admin/users", headers=headers)
    assert response.status_code == 403


def test_admin_list_users_success(client):
    """Test listing users as an admin user succeeds."""
    admin = create_test_user_helper("admin@example.com", "pass123", UserRole.admin)
    token = create_access_token(user_id=admin.id, email=admin.email, role=admin.role.value)
    
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/admin/users", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "users" in data
    assert data["count"] >= 1
    # Check that our admin is in the list
    emails = [u["email"] for u in data["users"]]
    assert "admin@example.com" in emails


def test_admin_set_user_role(client):
    """Test updating another user's role as an admin."""
    admin = create_test_user_helper("admin@example.com", "pass123", UserRole.admin)
    token = create_access_token(user_id=admin.id, email=admin.email, role=admin.role.value)
    
    target_user = create_test_user_helper("target@example.com", "pass123", UserRole.free)
    
    headers = {"Authorization": f"Bearer {token}"}
    response = client.patch(
        f"/admin/users/{target_user.id}/role",
        headers=headers,
        json={"role": "paid"}
    )
    assert response.status_code == 200
    assert response.json()["role"] == "paid"
    
    # Verify in DB
    updated = get_user_by_id(target_user.id)
    assert updated.role == UserRole.paid


def test_admin_set_user_role_not_found(client):
    """Test updating role of a non-existent user returns 404."""
    admin = create_test_user_helper("admin@example.com", "pass123", UserRole.admin)
    token = create_access_token(user_id=admin.id, email=admin.email, role=admin.role.value)
    
    headers = {"Authorization": f"Bearer {token}"}
    response = client.patch(
        "/admin/users/9999/role",
        headers=headers,
        json={"role": "paid"}
    )
    assert response.status_code == 404


def test_admin_deactivate_user(client):
    """Test soft-deactivating a user profile as an admin."""
    admin = create_test_user_helper("admin@example.com", "pass123", UserRole.admin)
    token = create_access_token(user_id=admin.id, email=admin.email, role=admin.role.value)
    
    target_user = create_test_user_helper("target@example.com", "pass123", UserRole.free)
    
    headers = {"Authorization": f"Bearer {token}"}
    response = client.delete(
        f"/admin/users/{target_user.id}",
        headers=headers
    )
    assert response.status_code == 204
    
    # Verify user is deactivated in DB
    updated = get_user_by_id(target_user.id)
    assert updated.is_active is False


def test_admin_deactivate_user_not_found(client):
    """Test deactivating a non-existent user returns 404."""
    admin = create_test_user_helper("admin@example.com", "pass123", UserRole.admin)
    token = create_access_token(user_id=admin.id, email=admin.email, role=admin.role.value)
    
    headers = {"Authorization": f"Bearer {token}"}
    response = client.delete(
        "/admin/users/9999",
        headers=headers
    )
    assert response.status_code == 404
