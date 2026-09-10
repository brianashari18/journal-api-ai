"""Unit test client LLM: provider detection + Gemini path (httpx di-mock)."""

import json


class _FakeResp:
    def __init__(self, data, status=200):
        self._data = data
        self._status = status

    def raise_for_status(self):
        if self._status != 200:
            raise RuntimeError(f"status {self._status}")

    def json(self):
        return self._data


def test_provider_auto_google_when_key_set(monkeypatch):
    import ollama_client

    monkeypatch.setenv("GOOGLE_API_KEY", "kunci")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert ollama_client._provider() == "google"


def test_provider_ollama_when_no_key(monkeypatch):
    import ollama_client

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert ollama_client._provider() == "ollama"


def test_provider_explicit(monkeypatch):
    import ollama_client

    monkeypatch.setenv("GOOGLE_API_KEY", "kunci")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert ollama_client._provider() == "ollama"


def test_google_chat_json_parses_candidate(monkeypatch):
    import ollama_client

    monkeypatch.setenv("GOOGLE_API_KEY", "kunci")
    captured = {}

    def fake_post(url, params=None, json=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["body"] = json
        return _FakeResp(
            {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
        )

    monkeypatch.setattr(ollama_client.httpx, "post", fake_post)
    out = ollama_client._google_chat_json("sys", "user prompt")
    assert json.loads(out) == {"ok": True}
    assert "gemini-2.5-flash" in captured["url"]
    assert captured["params"] == {"key": "kunci"}
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_google_chat_json_no_key_raises(monkeypatch):
    import ollama_client

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "google")
    try:
        ollama_client._google_chat_json("s", "u")
        assert False, "harus raise RuntimeError"
    except RuntimeError:
        pass