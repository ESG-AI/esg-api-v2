"""
routes/utils.py — Utility / metadata endpoints.

Endpoints:
  GET /scoring-rules   Return all scoring rules
  GET /categories      Return ESG categories and sub-categories
"""

from fastapi import APIRouter

from config import scoring_rules

router = APIRouter(tags=["Utilities"])


@router.get("/scoring-rules")
async def get_scoring_rules():
    """Return all scoring rules."""
    return scoring_rules


@router.get("/categories")
async def get_categories():
    """Return all available ESG categories."""
    categories: dict = {}

    for indicator_code, indicator in scoring_rules.items():
        category = indicator["types"]
        sub_category = indicator["sub-title"]

        if category not in categories:
            categories[category] = {}

        if sub_category not in categories[category]:
            categories[category][sub_category] = []

        categories[category][sub_category].append(
            {"code": indicator_code, "title": indicator["disclosure"]}
        )

    return categories
