def test_get_scoring_rules(client):
    """Test retrieving scoring rules endpoint."""
    response = client.get("/scoring-rules")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    # Ensure standard scoring rule properties exist in the response
    assert len(data) > 0
    # Check that at least one indicator contains expected fields
    first_key = list(data.keys())[0]
    assert "disclosure" in data[first_key]
    assert "types" in data[first_key]
    assert "sub-title" in data[first_key]


def test_get_categories(client):
    """Test retrieving categories endpoint."""
    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert len(data) > 0
    
    # Check that structure maps categories -> sub-categories -> indicator lists
    for category_name, subcategories in data.items():
        assert isinstance(subcategories, dict)
        for sub_title, indicators in subcategories.items():
            assert isinstance(indicators, list)
            for indicator in indicators:
                assert "code" in indicator
                assert "title" in indicator
