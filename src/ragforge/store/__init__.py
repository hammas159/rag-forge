"""Pick a store from config. Nothing above this line knows which one it got."""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .base import Store


@lru_cache(maxsize=1)
def get_store() -> Store:
    s = get_settings()
    backend = s.store.lower()
    if backend == "sqlite":
        from .sqlite import SqliteStore

        return SqliteStore(s.sqlite_path)
    if backend == "postgres":
        from .postgres import PostgresStore

        return PostgresStore()
    raise ValueError(f"Unknown STORE={backend!r}. Options: postgres, sqlite")


__all__ = ["Store", "get_store"]
