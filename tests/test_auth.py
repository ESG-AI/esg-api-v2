def test_register_user_success(client):
    """Test successful user registration."""
    response = client.post(
        "/auth/register",
        json={"email": "newuser@example.com", "password": "securepassword123"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert data["role"] == "free"
    assert data["is_active"] is True
    assert "id" in data


def test_register_user_duplicate_email(client):
    """Test registering a user with an already registered email."""
    # First registration
    client.post(
        "/auth/register",
        json={"email": "duplicate@example.com", "password": "password123"}
    )
    
    # Second registration with same email
    response = client.post(
        "/auth/register",
        json={"email": "duplicate@example.com", "password": "password123"}
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "An account with this email already exists."


def test_login_success(client):
    """Test successful login after registration."""
    # Register user first
    client.post(
        "/auth/register",
        json={"email": "loginuser@example.com", "password": "correctpassword"}
    )
    
    # Login
    response = client.post(
        "/auth/login",
        json={"email": "loginuser@example.com", "password": "correctpassword"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert isinstance(data["expires_in"], int)


def test_login_invalid_credentials(client):
    """Test login with incorrect password or non-existent email."""
    # Register a user
    client.post(
        "/auth/register",
        json={"email": "loginuser@example.com", "password": "correctpassword"}
    )
    
    # Login with wrong password
    response_wrong_password = client.post(
        "/auth/login",
        json={"email": "loginuser@example.com", "password": "wrongpassword"}
    )
    assert response_wrong_password.status_code == 401
    
    # Login with non-existent email
    response_wrong_email = client.post(
        "/auth/login",
        json={"email": "nonexistent@example.com", "password": "password123"}
    )
    assert response_wrong_email.status_code == 401


def test_get_current_user_profile(client):
    """Test retrieving authenticated user profile using token."""
    email = "profileuser@example.com"
    password = "password123"
    
    # Register and login to get access token
    client.post(
        "/auth/register",
        json={"email": email, "password": password}
    )
    login_response = client.post(
        "/auth/login",
        json={"email": email, "password": password}
    )
    access_token = login_response.json()["access_token"]
    
    # Access /auth/me with Bearer token
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/auth/me", headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == email
    assert data["role"] == "free"
    assert data["is_active"] is True
