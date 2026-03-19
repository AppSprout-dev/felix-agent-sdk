"""Felix memory systems.

Persistent knowledge storage, task pattern recognition, and context compression.
"""

from felix_agent_sdk.memory.compression import (
    CompressedContext,
    CompressionConfig,
    CompressionLevel,
    CompressionStrategy,
    ContextCompressor,
)
from felix_agent_sdk.memory.knowledge_store import (
    ConfidenceLevel,
    KnowledgeEntry,
    KnowledgeQuery,
    KnowledgeStore,
    KnowledgeType,
)
from felix_agent_sdk.memory.task_memory import (
    TaskComplexity,
    TaskExecution,
    TaskMemory,
    TaskMemoryQuery,
    TaskOutcome,
    TaskPattern,
)

__all__ = [
    # Knowledge store
    "KnowledgeStore",
    "KnowledgeEntry",
    "KnowledgeQuery",
    "KnowledgeType",
    "ConfidenceLevel",
    # Task memory
    "TaskMemory",
    "TaskPattern",
    "TaskExecution",
    "TaskMemoryQuery",
    "TaskOutcome",
    "TaskComplexity",
    # Compression
    "ContextCompressor",
    "CompressedContext",
    "CompressionConfig",
    "CompressionStrategy",
    "CompressionLevel",
]
