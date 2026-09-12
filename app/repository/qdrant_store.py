"""Qdrant store: collection, upsert (embed via Ollama), filter tanggal, similarity search."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    Range,
    VectorParams,
)

from app.services.llm_client import embed
from app.schemas.journal import EmotionItem, Entry
from app.core.config import settings

VECTOR_SIZE = 1024  # bge-m3


def client() -> QdrantClient:
    # check_compatibility=False: client 1.16 vs server 1.19 beda minor > 1,
    # tapi API yang dipakai stabil — warning-nya cuma noise.
    return QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)


def ensure_collection() -> None:
    c = client()
    if not c.collection_exists(settings.COLLECTION):
        c.create_collection(
            collection_name=settings.COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"[qdrant] Collection '{settings.COLLECTION}' dibuat (cosine, dim={VECTOR_SIZE}).")


def point_id(entry: Entry) -> str:
    # UUID stabil PER-ENTRI (tanggal+teks) -> upsert idempotent; 1 entri = 1 point,
    # emosi (maks 3) tersimpan di payload. Emosi TIDAK masuk hash (bukan identitas).
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"journal|{entry.date}|{entry.text}"))


def upsert_entries(entries: list[Entry]) -> None:
    c = client()
    ensure_collection()
    points = []
    for e in entries:
        points.append(
            PointStruct(
                id=point_id(e),
                vector=embed(f"{e.date} {e.day} {e.text}"),
                payload={
                    "date": e.date,
                    "date_ts": int(e.date.replace("-", "")),  # YYYYMMDD -> filter range numerik
                    "day": e.day,
                    "text": e.text,
                    "emotions": [ei.model_dump() for ei in e.emotions],
                },
            )
        )
    c.upsert(collection_name=settings.COLLECTION, points=points)


def _to_entry(payload: dict) -> Entry:
    return Entry(
        date=payload["date"],
        day=payload.get("day", ""),
        text=payload["text"],
        emotions=[EmotionItem(**ei) for ei in payload.get("emotions", [])],
    )


def retrieve_by_date(start: str, end: str) -> list[Entry]:
    """Retrieval RAG: filter skalar (tanggal) via payload. Semua entri dalam rentang."""
    c = client()
    records, _ = c.scroll(
        collection_name=settings.COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="date_ts",
                    range=Range(gte=int(start.replace("-", "")), lte=int(end.replace("-", ""))),
                )
            ]
        ),
        limit=1000,
        with_payload=True,
    )
    return [_to_entry(r.payload) for r in records]


def search_similar(text: str, limit: int = 5) -> list[Entry]:
    """Retrieval RAG klasik: similarity search di vektor."""
    c = client()
    hits = c.query_points(
        collection_name=settings.COLLECTION,
        query=embed(text),
        limit=limit,
        with_payload=True,
    ).points
    return [_to_entry(h.payload) for h in hits]