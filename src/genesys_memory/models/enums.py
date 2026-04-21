from enum import Enum


class MemoryStatus(str, Enum):
    TAGGED = "tagged"
    ACTIVE = "active"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    CORE = "core"
    DORMANT = "dormant"
    PRUNED = "pruned"


class EdgeType(str, Enum):
    CAUSED_BY = "caused_by"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived_from"
    RELATED_TO = "related_to"
    TEMPORAL_SEQUENCE = "temporal_sequence"


CAUSAL_EDGE_TYPES = {EdgeType.CAUSED_BY, EdgeType.SUPPORTS, EdgeType.DERIVED_FROM}


class ReactivationPattern(str, Enum):
    BURST = "burst"
    STEADY = "steady"
    SINGLE = "single"
