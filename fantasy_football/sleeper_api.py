"""
Read-only Sleeper API helpers used by the fantasy-football pages.

Sleeper's public API does not require authentication.  Keep API access in this
module so the model/ranking code remains independent of Sleeper.
"""

from __future__ import annotations

from typing import Any

import re

import requests


BASE_URL = "https://api.sleeper.app/v1"
DEFAULT_TIMEOUT = 15


def extract_draft_id(value: str) -> str:
    """
    Accept either a raw Sleeper draft ID or a Sleeper draftboard URL.

    Examples:
        1234567890123456789
        https://sleeper.com/draft/nfl/1234567890123456789
        https://sleeper.app/draft/nfl/1234567890123456789
    """
    value = str(value or "").strip()
    if not value:
        raise ValueError("Sleeper draft ID or draft URL is required.")

    match = re.search(r"/draft/(?:nfl/)?(\d+)", value)
    if match:
        return match.group(1)

    if value.isdigit():
        return value

    raise ValueError(
        "Could not find a Sleeper draft ID. Paste the numeric draft ID or "
        "the full Sleeper draftboard URL."
    )


def _get_json(path: str) -> Any:
    response = requests.get(
        f"{BASE_URL}{path}",
        timeout=DEFAULT_TIMEOUT,
    )

    if response.status_code == 404:
        raise ValueError(f"Sleeper resource was not found: {path}")

    response.raise_for_status()
    return response.json()


def get_league(league_id: str) -> dict:
    """Return a Sleeper league object, including scoring_settings."""
    league_id = str(league_id).strip()
    if not league_id:
        raise ValueError("Sleeper league ID is required.")

    league = _get_json(f"/league/{league_id}")

    if not isinstance(league, dict) or not league.get("league_id"):
        raise ValueError("Sleeper returned an invalid league response.")

    return league


def get_league_drafts(league_id: str) -> list[dict]:
    """Return all Sleeper drafts for a league, newest first."""
    league_id = str(league_id).strip()
    if not league_id:
        raise ValueError("Sleeper league ID is required.")

    drafts = _get_json(f"/league/{league_id}/drafts")
    if not isinstance(drafts, list):
        raise ValueError("Sleeper returned an invalid draft list.")
    return drafts


def get_draft(draft_id: str) -> dict:
    """Return one Sleeper draft object."""
    draft_id = str(draft_id).strip()
    if not draft_id:
        raise ValueError("Sleeper draft ID is required.")

    draft = _get_json(f"/draft/{draft_id}")
    if not isinstance(draft, dict) or not draft.get("draft_id"):
        raise ValueError("Sleeper returned an invalid draft response.")
    return draft


def get_draft_picks(draft_id: str) -> list[dict]:
    """Return all completed picks in a Sleeper draft."""
    draft_id = str(draft_id).strip()
    if not draft_id:
        raise ValueError("Sleeper draft ID is required.")

    picks = _get_json(f"/draft/{draft_id}/picks")
    if not isinstance(picks, list):
        raise ValueError("Sleeper returned an invalid picks response.")

    return sorted(
        picks,
        key=lambda pick: int(pick.get("pick_no") or 0),
    )
