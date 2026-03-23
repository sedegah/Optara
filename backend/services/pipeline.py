from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecognitionResult:
    label: str
    confidence: float
    user_id: int | None = None
    user_name: str | None = None
