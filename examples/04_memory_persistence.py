#!/usr/bin/env python3
"""Memory persistence with KnowledgeStore and TaskMemory.

Demonstrates the pluggable memory backend: storing knowledge entries,
querying by type/domain, recording task executions, and getting
strategy recommendations based on historical patterns.

Usage:
    python examples/04_memory_persistence.py
"""

from felix_agent_sdk.memory import (
    ConfidenceLevel,
    KnowledgeQuery,
    KnowledgeStore,
    KnowledgeType,
    TaskMemory,
)
from felix_agent_sdk.memory.backends.sqlite import SQLiteBackend
from felix_agent_sdk.memory.task_memory import TaskComplexity, TaskOutcome


def knowledge_store_demo():
    """Demonstrate KnowledgeStore CRUD operations."""
    print("=== KnowledgeStore Demo ===\n")

    # Use in-memory backend (pass db_path for persistence)
    backend = SQLiteBackend()
    store = KnowledgeStore(backend=backend)

    # Add knowledge entries
    kid1 = store.add_entry(
        knowledge_type=KnowledgeType.AGENT_INSIGHT,
        content={"text": "Solar capacity grew 40% year-over-year in 2025"},
        confidence_level=ConfidenceLevel.HIGH,
        source_agent="research-001",
        domain="energy",
        tags=["solar", "growth"],
    )
    kid2 = store.add_entry(
        knowledge_type=KnowledgeType.TASK_RESULT,
        content={"text": "Battery costs declined 85% since 2010"},
        confidence_level=ConfidenceLevel.VERIFIED,
        source_agent="analysis-001",
        domain="energy",
    )
    kid3 = store.add_entry(
        knowledge_type=KnowledgeType.FAILURE_ANALYSIS,
        content={"text": "Grid intermittency remains unsolved at scale"},
        confidence_level=ConfidenceLevel.MEDIUM,
        source_agent="critic-001",
        domain="energy",
    )

    print(f"Added 3 entries: {kid1[:8]}..., {kid2[:8]}..., {kid3[:8]}...")

    # Query by domain
    energy = store.get_entries_by_domain("energy")
    print(f"Entries in 'energy' domain: {len(energy)}")

    # Query with filters
    high_conf = store.search(
        KnowledgeQuery(min_confidence=ConfidenceLevel.HIGH)
    )
    print(f"High+ confidence entries: {len(high_conf)}")

    # Add a relationship
    store.add_relationship(kid1, kid2, "supports", confidence=0.9)
    rels = store.get_relationships(kid1)
    print(f"Relationships for {kid1[:8]}: {len(rels)}")

    # Summary
    summary = store.get_summary()
    print(f"Store summary: {summary}")

    backend.close()


def task_memory_demo():
    """Demonstrate TaskMemory pattern recognition."""
    print("\n=== TaskMemory Demo ===\n")

    backend = SQLiteBackend()
    memory = TaskMemory(backend=backend)

    # Record successful executions
    memory.record_task_execution(
        task_description="Analyse renewable energy market trends",
        task_type="market_analysis",
        complexity=TaskComplexity.MODERATE,
        outcome=TaskOutcome.SUCCESS,
        duration=15.0,
        agents_used=["research", "analysis"],
        strategies_used=["parallel_research", "deep_dive"],
        context_size=500,
    )
    memory.record_task_execution(
        task_description="Analyse renewable energy policy impacts",
        task_type="market_analysis",
        complexity=TaskComplexity.MODERATE,
        outcome=TaskOutcome.SUCCESS,
        duration=12.0,
        agents_used=["research", "analysis", "critic"],
        strategies_used=["parallel_research"],
        context_size=600,
    )

    # Get strategy recommendation
    rec = memory.recommend_strategy(
        "Analyse renewable energy investment opportunities",
        "market_analysis",
        TaskComplexity.MODERATE,
    )
    print(f"Recommended strategies: {rec['strategies']}")
    print(f"Recommended agents: {rec['agents']}")
    print(f"Success probability: {rec['success_probability']:.2f}")
    print(f"Estimated duration: {rec['estimated_duration']:.1f}s")

    # Summary
    summary = memory.get_memory_summary()
    print(f"Memory summary: {summary}")

    backend.close()


def main():
    knowledge_store_demo()
    task_memory_demo()


if __name__ == "__main__":
    main()
