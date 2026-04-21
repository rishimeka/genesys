"""Centralized engine configuration.

All tunable thresholds in one place. Every value can be overridden via
environment variable. Defaults match the values validated against the
LoCoMo benchmark (89.9%) and the neuroscience literature (ACT-R, Rescorla-Wagner).
"""
from __future__ import annotations

import os


def _float(key: str, default: str) -> float:
    return float(os.getenv(key, default))


def _int(key: str, default: str) -> int:
    return int(os.getenv(key, default))


# ---------------------------------------------------------------------------
# ACT-R Scoring (scoring.py)
# ---------------------------------------------------------------------------
ACTR_DECAY_EXPONENT = _float("GENESYS_ACTR_DECAY", "0.5")
RELEVANCE_VECTOR_WEIGHT = _float("GENESYS_RELEVANCE_VECTOR_WEIGHT", "0.7")
RELEVANCE_KEYWORD_WEIGHT = _float("GENESYS_RELEVANCE_KEYWORD_WEIGHT", "0.3")
MIN_CONNECTIVITY = _float("GENESYS_MIN_CONNECTIVITY", "0.1")

# ---------------------------------------------------------------------------
# Status Transitions (transitions.py)
# ---------------------------------------------------------------------------
TAGGED_EXPIRE_HOURS = _int("GENESYS_TAGGED_EXPIRE_HOURS", "24")
ACTIVE_TO_EPISODIC_THRESHOLD = _float("GENESYS_ACTIVE_EPISODIC_THRESHOLD", "0.6")
ACTIVE_TO_EPISODIC_SESSIONS = _int("GENESYS_ACTIVE_EPISODIC_SESSIONS", "3")
DORMANCY_THRESHOLD = _float("GENESYS_DORMANCY_THRESHOLD", "0.15")
DORMANCY_DAYS = _int("GENESYS_DORMANCY_DAYS", "90")
DORMANCY_MAX_REACTIVATIONS = _int("GENESYS_DORMANCY_MAX_REACTIVATIONS", "3")

# ---------------------------------------------------------------------------
# Active Forgetting (forgetting.py)
# ---------------------------------------------------------------------------
FORGETTING_THRESHOLD = _float("GENESYS_FORGETTING_THRESHOLD", "0.01")

# ---------------------------------------------------------------------------
# Core Memory Promotion (promoter.py)
# ---------------------------------------------------------------------------
CORE_THRESHOLD = _float("GENESYS_CORE_THRESHOLD", "0.55")
CORE_ACTIVATION_WEIGHT = _float("GENESYS_CORE_ACTIVATION_WEIGHT", "0.4")
CORE_HUB_WEIGHT = _float("GENESYS_CORE_HUB_WEIGHT", "0.3")
CORE_SCHEMA_WEIGHT = _float("GENESYS_CORE_SCHEMA_WEIGHT", "0.2")
CORE_STABILITY_WEIGHT = _float("GENESYS_CORE_STABILITY_WEIGHT", "0.1")
AUTO_PROMOTE_CATEGORIES = [
    c.strip()
    for c in os.getenv(
        "GENESYS_AUTO_PROMOTE_CATEGORIES",
        "professional,educational,family,location",
    ).split(",")
]
HUB_SCORE_CAP = _float("GENESYS_HUB_SCORE_CAP", "3.0")
SCHEMA_NEIGHBOR_CAP = _int("GENESYS_SCHEMA_NEIGHBOR_CAP", "10")
STABILITY_CAP = _float("GENESYS_STABILITY_CAP", "3.0")

# ---------------------------------------------------------------------------
# Cascade Reactivation (reactivation.py)
# ---------------------------------------------------------------------------
CASCADE_DEPTH = _int("GENESYS_CASCADE_DEPTH", "2")
CASCADE_DECAY_FACTOR = _float("GENESYS_CASCADE_DECAY_FACTOR", "0.3")
DORMANT_REVIVAL_THRESHOLD = _float("GENESYS_DORMANT_REVIVAL_THRESHOLD", "0.1")

# ---------------------------------------------------------------------------
# Ingestion Limits
# ---------------------------------------------------------------------------
MAX_INGEST_FILE_MB = _int("GENESYS_MAX_INGEST_FILE_MB", "100")
