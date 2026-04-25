"""System operation modes and related helpers."""

from __future__ import annotations

from enum import Enum


class AssistantMode(str, Enum):
    SCENE_DESCRIPTION = "scene_description"
    NAVIGATION = "navigation"
    OBSTACLE_AWARENESS = "obstacle_awareness"
    OBJECT_FINDER = "object_finder"
    TEXT_READING = "text_reading"
    PAUSED = "paused"
