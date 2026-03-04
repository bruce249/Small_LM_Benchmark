"""SQLAlchemy async engine & session factory.

Engine and session factory are created lazily so the app can start
even when PostgreSQL / asyncpg are not available (e.g. demo mode).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_engine = None
_session_factory = None


def _init_db():
    """Lazily create the engine and session factory."""
    global _engine, _session_factory
    if _engine is not None:
        return

    from sqlalchemy.ext.asyncio import (
        AsyncSession as _AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from arena.config import get_settings

    _settings = get_settings()
    _engine = create_async_engine(
        _settings.database_url,
        echo=_settings.database_echo,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )
    _session_factory = async_sessionmaker(
        _engine,
        class_=_AsyncSession,
        expire_on_commit=False,
    )


async def get_db() -> "AsyncSession":  # type: ignore[misc]
    """Yield an async DB session – use as a FastAPI dependency."""
    _init_db()
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
