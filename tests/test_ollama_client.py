"""Unit test client LLM: provider detection + Gemini path (httpx di-mock)."""

import json
from app.services import llm_client as ollama_client
from app.core.config import settings


class _FakeResp:
    def __init__(self, data, status=200):
        self._data = data
        self._status = status
        self.status_code = status

    def raise_for_status(self):
        if self._status != 200:
            raise RuntimeError(f"status {self._status}")

    def json(self):
        return self._data


def test_provider_auto_google_when_key_set(monkeypatch):
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "kunci")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "")
    assert ollama_client._chat_provider() == "google"


def test_provider_ollama_when_no_key(monkeypatch):
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "")
    assert ollama_client._chat_provider() == "ollama"


def test_provider_explicit(monkeypatch):
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "kunci")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    assert ollama_client._chat_provider() == "ollama"


def test_google_chat_json_parses_candidate(monkeypatch):
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "kunci")
    captured = {}

    def fake_post(url, params=None, json=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["body"] = json
        return _FakeResp(
            {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
        )

    monkeypatch.setattr(ollama_client.httpx, "post", fake_post)
    out = ollama_client._google_chat_json("sys", "user prompt", 0.7)
    assert json.loads(out) == {"ok": True}
    assert "gemini-2.5-flash" in captured["url"]
    assert captured["params"] == {"key": "kunci"}
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert captured["body"]["generationConfig"]["temperature"] == 0.7


def test_google_chat_json_no_key_raises(monkeypatch):
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "google")
    try:
        ollama_client._google_chat_json("s", "u", 0.7)
        assert False, "harus raise RuntimeError"
    except RuntimeError:
        pass