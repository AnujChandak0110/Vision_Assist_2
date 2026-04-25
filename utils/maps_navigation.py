"""
Map-based navigation helper using Google Directions API (optional).
"""

from __future__ import annotations

import os
from typing import Optional

import requests
from requests import RequestException


def get_next_navigation_instruction(origin: str, destination: str) -> str:
    """
    Return concise next-step navigation instruction.

    Requires MAPS_API_KEY in environment.
    """
    api_key = os.getenv("MAPS_API_KEY", "").strip() or os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        return ""

    try:
        response = requests.get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params={
                "origin": origin,
                "destination": destination,
                "mode": "walking",
                "key": api_key,
            },
            timeout=6,
        )
        response.raise_for_status()
        payload = response.json()

        routes = payload.get("routes", [])
        if not routes:
            return ""

        legs = routes[0].get("legs", [])
        if not legs or not legs[0].get("steps"):
            return ""

        step_html = legs[0]["steps"][0].get("html_instructions", "Move ahead carefully.")
        clean = (
            step_html.replace("<b>", "")
            .replace("</b>", "")
            .replace("<div style=\"font-size:0.9em\">", " ")
            .replace("</div>", "")
        )
        clean = " ".join(clean.split())
        return clean[:160]

    except RequestException:
        return ""
