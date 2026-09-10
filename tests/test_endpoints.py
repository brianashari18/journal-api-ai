"""Integration test wiring endpoint (butuh Qdrant + Ollama bge-m3; auto-skip kalau mati)."""

import json

import httpx
import pytest

import main as app_module  # noqa: E402
import summarizer as summarizer_module  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _infra_up() -> bool:
    try:
        tags = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2).json().get("models", [])
        qdrant = httpx.get("http://127.0.0.1:6333/healthz", timeout=2).status_code
        return qdrant == 200 and any(m["name"].startswith("bge-m3") for m in tags)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _infra_up(), reason="Qdrant/Ollama bge-m3 tidak hidup")


@pytest.fixture
def client(monkeypatch):
    def fake_chat_json(system: str, user: str) -> str:
        if "PERTANYAAN" in user:
            return json.dumps(
                {"answer": "Paling cemas 2026-09-01 karena deadline.", "sources": ["2026-09-01"]}
            )
        return json.dumps(
            {
                "patterns": [],
                "changes": [],
                "reflection_question": "Apa yang membantu kamu lebih tenang minggu ini?",
            }
        )

    monkeypatch.setattr("main.chat_json", fake_chat_json)
    monkeypatch.setattr("summarizer.chat_json", fake_chat_json)
    return TestClient(app_module.app)


ENTRY = {
    "date": "2026-09-01",
    "day": "Selasa",
    "emotions": [{"emotion": "cemas", "intensity": 8}],
    "text": "Deadline proyek minggu depan bikin gak bisa tidur.",
}


def test_entries_ok(client):
    resp = client.post(
        "/ai/entries",
        json={
            "entries": [
                {
                    "date": "2026-09-05",
                    "text": "Baru nyoba guided journaling.",
                    "emotions": [{"emotion": "tenang", "intensity": 6}],
                }
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert len(body["entry_ids"]) == 1


def test_entries_empty_400(client):
    resp = client.post("/ai/entries", json={"entries": []})
    assert resp.status_code == 400


def test_summarize_ok(client):
    resp = client.post(
        "/ai/summarize",
        json={"start_date": "2026-09-01", "end_date": "2026-09-07", "entries": [ENTRY]},
    )
    assert resp.status_code == 200
    body = resp.json()
    # integrasi: Qdrant shared, bisa ada data lain dalam rentang -> assert >= 1
    assert body["period"]["entry_count"] >= 1
    assert "cemas" in [o["emotion"] for o in body["emotion_overview"]]
    assert any(o["emotion"] == "cemas" and o.get("intensity_mean") for o in body["emotion_overview"])
    assert body["timeline"] and body["timeline"][0]["entry_id"]
    # LLM dikosongkan di stub -> semua kandidat di-template, evidence tetap ada
    assert body["changes"] and body["changes"][0]["supporting_entry_ids"]
    assert body["presentation"]["header"]["title"] == "Minggu Emosimu"
    assert body["presentation"]["timeline"]


def test_query_ok(client):
    resp = client.post("/ai/query", json={"question": "kapan gue paling cemas?"})
    assert resp.status_code == 200
    assert resp.json()["sources"] == ["2026-09-01"]


def test_summarize_empty_entries_404(client):
    resp = client.post(
        "/ai/summarize",
        json={"start_date": "2025-01-01", "end_date": "2025-01-07", "entries": []},
    )
    assert resp.status_code == 404