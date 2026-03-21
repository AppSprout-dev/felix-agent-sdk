"""Shared text processing utilities."""

from __future__ import annotations

# Common English stopwords used by keyword extraction across the SDK.
# Used by: memory/compression.py, memory/task_memory.py, spawning/content_analyzer.py
STOPWORDS: frozenset[str] = frozenset({
    # Articles / determiners
    "the", "a", "an", "this", "that", "these", "those",
    # Pronouns
    "it", "its", "he", "him", "his", "she", "her", "you", "your",
    "our", "we", "they", "them",
    # Be-verbs
    "is", "are", "was", "were", "be", "been", "being",
    # Have-verbs
    "have", "has", "had",
    # Do-verbs
    "do", "does", "did",
    # Modals
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "need", "must",
    # Prepositions
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "about", "over",
    # Conjunctions
    "and", "or", "but", "if", "so", "than",
    # Adverbs / misc
    "not", "no", "too", "very", "just", "also", "here", "now", "how",
    "all", "one", "two", "new", "old", "much", "many", "some", "such",
    "well", "long", "good", "time", "make", "take", "come", "like",
    "when", "where", "know", "want",
    # Short common words (from memory module lists)
    "out", "day", "get", "see", "who", "boy", "man", "use", "way",
    "oil", "sit", "set", "run",
})
