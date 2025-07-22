"""Utilities for querying GraphRAG indexes."""
from __future__ import annotations

import asyncio
from pathlib import Path

from graphrag.cli import query as rag_query


def search(query: str, *, root_dir: str = "graphrag_project", config: str | None = None, data_dir: str | None = None, level: int = 2) -> str:
    """Return GraphRAG search result or error message."""
    try:
        response, _ = rag_query.run_global_search(
            config_filepath=Path(config) if config else None,
            data_dir=Path(data_dir) if data_dir else None,
            root_dir=Path(root_dir),
            community_level=level,
            dynamic_community_selection=False,
            response_type="Multiple Paragraphs",
            streaming=False,
            query=query,
            verbose=False,
        )
        if isinstance(response, str):
            return response
        return str(response)
    except Exception as e:  # pragma: no cover - external
        return f"[GraphRAG error: {e}]"
