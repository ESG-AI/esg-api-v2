import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from services.evaluation import (
    chunk_dict,
    build_batched_prompt,
    evaluate_indicator,
    evaluate_indicator_batch,
)

# ---------------------------------------------------------------------------
# Test Pure Utilities
# ---------------------------------------------------------------------------

def test_chunk_dict():
    """Test chunking a dictionary into smaller batches."""
    input_dict = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    chunks = list(chunk_dict(input_dict, 2))
    
    assert len(chunks) == 3
    assert chunks[0] == {"A": 1, "B": 2}
    assert chunks[1] == {"C": 3, "D": 4}
    assert chunks[2] == {"E": 5}


def test_build_batched_prompt():
    """Test generating batched prompt string structure."""
    indicator_batch = {
        "GRI-302-1": {
            "disclosure": "Energy consumption within the organization",
            "description": "Report energy consumption details",
            "criteria": {"0": "No info", "4": "Complete details"},
            "keywords": ["energy", "electricity"]
        }
    }
    system, user = build_batched_prompt(indicator_batch, "Some company report text")
    
    assert "You are an expert ESG analyst using GRI standards" in system
    assert "GRI-302-1" in system
    assert "Energy consumption within the organization" in system
    assert "DOCUMENT TEXT:" in user
    assert "Some company report text" in user


# ---------------------------------------------------------------------------
# Test Async Evaluation with Mocks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_indicator_batch_success():
    """Test successful batch indicator evaluation using a mocked client."""
    mock_client = AsyncMock()
    
    # Setup mock API response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "results": [
                {
                    "indicator_code": "GRI-302-1",
                    "score": 3,
                    "reasoning": "Company discloses high electricity usage."
                }
            ]
        })))
    ]
    mock_response.usage = MagicMock(total_tokens=150, prompt_tokens=100, completion_tokens=50)
    mock_client.chat.completions.create.return_value = mock_response
    
    indicator_batch = {"GRI-302-1": {
        "disclosure": "Energy", "description": "Desc", "criteria": {}, "keywords": []
    }}
    
    results, tokens = await evaluate_indicator_batch(mock_client, "Sample Report", indicator_batch)
    
    assert len(results) == 1
    assert results[0]["indicator_code"] == "GRI-302-1"
    assert results[0]["score"] == 3
    assert results[0]["reasoning"] == "Company discloses high electricity usage."
    assert tokens["total_tokens"] == 150
    assert tokens["prompt_tokens"] == 100
    assert tokens["response_tokens"] == 50


@pytest.mark.asyncio
async def test_evaluate_indicator_batch_parse_failure():
    """Test handler behavior when the AI response returns invalid JSON."""
    mock_client = AsyncMock()
    
    # Setup mock with invalid JSON content
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Invalid JSON response from model"))
    ]
    mock_response.usage = MagicMock(total_tokens=50, prompt_tokens=40, completion_tokens=10)
    mock_client.chat.completions.create.return_value = mock_response
    
    indicator_batch = {"GRI-302-1": {
        "disclosure": "Energy", "description": "Desc", "criteria": {}, "keywords": []
    }}
    
    results, tokens = await evaluate_indicator_batch(mock_client, "Sample Report", indicator_batch)
    
    # Should safely fail and return empty results instead of raising exception
    assert results == []
    assert tokens["total_tokens"] == 50


@pytest.mark.asyncio
@patch("services.evaluation.openai_client.chat.completions.create", new_callable=AsyncMock)
async def test_evaluate_indicator_success(mock_chat_create):
    """Test single indicator evaluation using the globally configured client."""
    # Setup mock API response for single indicator evaluation format: "Score\nReasoning"
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="2\nReasoning explanation for score 2."))
    ]
    mock_response.usage = MagicMock(total_tokens=120, prompt_tokens=80, completion_tokens=40)
    mock_chat_create.return_value = mock_response
    
    indicator = {
        "disclosure": "Water",
        "description": "Water details",
        "criteria": {"0": "No water details", "4": "Full details"},
        "keywords": ["water", "consumption"]
    }
    
    score, reasoning, tokens = await evaluate_indicator(
        "Water consumption is 500 liters.",
        "GRI-303-1",
        indicator,
        "gpt-4o-mini"
    )
    
    assert score == 2
    assert reasoning == "Reasoning explanation for score 2."
    assert tokens["total_tokens"] == 120
