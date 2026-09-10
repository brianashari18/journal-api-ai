import json

import summarizer
from schemas import EmotionItem, Entry


def _e(date, emotions, text):
    return Entry(date=date, day="", emotions=[EmotionItem(**x) for x in emotions], text=text)


def test_extract_context_keyword():
    assert "revision" in summarizer.extract_context("got many revisions from lecturer")
    assert "university" in summarizer.extract_context("lecturer gave revisions")
    assert summarizer.extract_context("no keyword here") == []


def test_build_structured_overview_and_timeline():
    entries = [
        _e("2026-09-01", [{"emotion": "cemas", "intensity": 8}], "deadline proyek bikin cemas"),
        _e(
            "2026-09-02",
            [{"emotion": "cemas", "intensity": 6}, {"emotion": "senang", "intensity": 7}],
            "deadline masih ngejar",
        ),
        _e("2026-09-03", [{"emotion": "senang", "intensity": 9}], "ketemu teman"),
    ]
    s = summarizer.build_structured(entries)
    assert len(s["emotion_overview"]) == 2
    cemas = next(o for o in s["emotion_overview"] if o.emotion == "cemas")
    senang = next(o for o in s["emotion_overview"] if o.emotion == "senang")
    assert cemas.entry_count == 2
    assert senang.entry_count == 2
    assert cemas.intensity_mean == 7.0
    assert cemas.intensity_max == 8
    assert cemas.peak_date == "2026-09-01"
    assert cemas.peak_entry_id == s["timeline"][0].entry_id
    assert [t.date for t in s["timeline"]] == ["2026-09-01", "2026-09-02", "2026-09-03"]
    # emosi utama entri 2 = senang (7 > 6)
    assert s["timeline"][1].emotion == "senang"
    assert all(t.entry_id for t in s["timeline"])


def test_multi_emotion_counts_in_both_labels():
    entries = [
        _e("2026-09-01", [{"emotion": "cemas", "intensity": 8}, {"emotion": "capek", "intensity": 6}], "x"),
    ]
    s = summarizer.build_structured(entries)
    assert {o.emotion for o in s["emotion_overview"]} == {"cemas", "capek"}
    for o in s["emotion_overview"]:
        assert o.entry_count == 1


def test_candidate_patterns_require_two_entries():
    entries = [
        _e("2026-09-01", [{"emotion": "frustrated", "intensity": 8}], "got many revisions"),
        _e("2026-09-02", [{"emotion": "frustrated", "intensity": 6}], "still many revisions"),
        _e("2026-09-03", [{"emotion": "cemas", "intensity": 7}], "revisi doang cemas"),
    ]
    s = summarizer.build_structured(entries)
    fr = [p for p in s["candidate_patterns"] if p["emotion"] == "frustrated"]
    assert len(fr) == 1
    assert fr[0]["context"] == "revision"
    assert len(fr[0]["entry_ids"]) == 2


def test_single_entry_not_a_pattern():
    entries = [_e("2026-10-05", [{"emotion": "cemas", "intensity": 7}], "presentasi besok bikin cemas")]
    s = summarizer.build_structured(entries)
    assert s["candidate_patterns"] == []


def test_primary_emotion_used_for_changes():
    entries = [
        _e("2026-10-05", [{"emotion": "cemas", "intensity": 9}], "presentasi besok"),
        _e("2026-10-06", [{"emotion": "relieved", "intensity": 9}, {"emotion": "cemas", "intensity": 3}], "presentasi lancar"),
    ]
    s = summarizer.build_structured(entries)
    assert len(s["candidate_changes"]) == 1
    assert s["candidate_changes"][0]["from_emotion"] == "cemas"
    assert s["candidate_changes"][0]["to_emotion"] == "relieved"


def test_generate_narrative_uses_candidate_ids(monkeypatch):
    fake = json.dumps(
        {
            "patterns": [
                {"candidate_id": "p0", "description": "Frustrated appeared in two entries related to revisions."}
            ],
            "changes": [{"candidate_id": "c0", "description": "Frustrated was followed by cemas."}],
            "reflection_question": "What helped you move from frustration to relief?",
        }
    )
    monkeypatch.setattr("summarizer.chat_json", lambda system, user: fake)
    entries = [
        _e("2026-09-01", [{"emotion": "frustrated", "intensity": 8}], "got many revisions"),
        _e("2026-09-02", [{"emotion": "frustrated", "intensity": 6}], "still many revisions"),
        _e("2026-09-03", [{"emotion": "cemas", "intensity": 7}], "x"),
        _e("2026-09-04", [{"emotion": "relieved", "intensity": 8}], "y"),
    ]
    s = summarizer.build_structured(entries)
    p, c, r = summarizer.generate_narrative(s)
    assert p[0].description.startswith("Frustrated appeared in two entries")
    assert p[0].supporting_entry_ids == s["candidate_patterns"][0]["entry_ids"]
    assert c[0].supporting_entry_ids == s["candidate_changes"][0]["entry_ids"]
    assert len(c) == 2
    assert r.startswith("What")


def test_assemble_rejects_causal_and_out_of_candidate(monkeypatch):
    fake = json.dumps(
        {
            "patterns": [
                {
                    "candidate_id": "p0",
                    "description": "Frustrated appeared in entries related to revision because of deadlines.",
                }
            ],
            "changes": [{"candidate_id": "c0", "description": "Surprise turned into joy."}],
            "reflection_question": "Apa yang membuat perasaanmu berubah?",
        }
    )
    monkeypatch.setattr("summarizer.chat_json", lambda system, user: fake)
    entries = [
        _e("2026-09-01", [{"emotion": "frustrated", "intensity": 8}], "got many revisions"),
        _e("2026-09-02", [{"emotion": "frustrated", "intensity": 6}], "still many revisions"),
        _e("2026-09-03", [{"emotion": "cemas", "intensity": 7}], "x"),
    ]
    s = summarizer.build_structured(entries)
    p, c, r = summarizer.generate_narrative(s)
    assert p[0].description.startswith("Frustrated muncul di 2 entri")
    assert c[0].description == "Kamu menulis tentang frustrated, lalu cemas."
    assert c[0].supporting_entry_ids == s["candidate_changes"][0]["entry_ids"]
    assert r == summarizer._REFLECTION_FALLBACK


def test_reflection_ok_passes_non_causal():
    assert summarizer._reflection_ok("What helped you feel calmer this week?")
    assert not summarizer._reflection_ok("Apa yang membuat kamu tenang?")
    assert not summarizer._reflection_ok("")


def test_generate_narrative_drops_unknown_candidate_id(monkeypatch):
    fake = json.dumps(
        {
            "patterns": [{"candidate_id": "zzz", "description": "Invented pattern not in evidence."}],
            "changes": [],
            "reflection_question": "pertanyaan?",
        }
    )
    monkeypatch.setattr("summarizer.chat_json", lambda system, user: fake)
    entries = [
        _e("2026-09-01", [{"emotion": "frustrated", "intensity": 8}], "got many revisions"),
        _e("2026-09-02", [{"emotion": "frustrated", "intensity": 6}], "still many revisions"),
    ]
    s = summarizer.build_structured(entries)
    p, c, r = summarizer.generate_narrative(s)
    assert p[0].description.startswith("Frustrated muncul di 2 entri")
    assert p[0].supporting_entry_ids == s["candidate_patterns"][0]["entry_ids"]


def test_generate_narrative_empty_llm_templates_everything(monkeypatch):
    fake = json.dumps({"patterns": [], "changes": [], "reflection_question": ""})
    monkeypatch.setattr("summarizer.chat_json", lambda system, user: fake)
    entries = [
        _e("2026-09-01", [{"emotion": "frustrated", "intensity": 8}], "got many revisions"),
        _e("2026-09-02", [{"emotion": "frustrated", "intensity": 6}], "still many revisions"),
        _e("2026-09-03", [{"emotion": "cemas", "intensity": 7}], "x"),
    ]
    s = summarizer.build_structured(entries)
    p, c, r = summarizer.generate_narrative(s)
    assert len(p) == 1 and p[0].description.startswith("Frustrated muncul di 2 entri")
    assert c and all(ch.supporting_entry_ids for ch in c)
    assert r == summarizer._REFLECTION_FALLBACK


def test_generate_narrative_fallback_on_bad_json(monkeypatch):
    monkeypatch.setattr("summarizer.chat_json", lambda system, user: "bukan json")
    entries = [
        _e("2026-09-01", [{"emotion": "frustrated", "intensity": 8}], "got many revisions"),
        _e("2026-09-02", [{"emotion": "frustrated", "intensity": 6}], "still many revisions"),
    ]
    s = summarizer.build_structured(entries)
    p, c, r = summarizer.generate_narrative(s)
    assert p[0].description.startswith("Frustrated muncul di 2 entri")
    assert p[0].supporting_entry_ids


def test_build_presentation_intensity_and_emotions():
    entries = [
        _e("2026-10-05", [{"emotion": "anxious", "intensity": 8}], "I have a presentation tomorrow and I keep thinking about it."),
        _e("2026-10-06", [{"emotion": "relieved", "intensity": 9}], "The presentation went well. I feel really relieved."),
        _e("2026-10-07", [{"emotion": "frustrated", "intensity": 7}], "I got so many revisions from my lecturer."),
        _e("2026-10-08", [{"emotion": "frustrated", "intensity": 8}, {"emotion": "capek", "intensity": 5}], "Still frustrated with the revisions."),
    ]
    s = summarizer.build_structured(entries)
    pres = summarizer.build_presentation(s, "2026-10-05", "2026-10-11", "Q?")
    assert pres.header.title == "Minggu Emosimu"
    assert pres.header.period_label == "Okt 5 – Okt 11"
    assert pres.overview_labels == ["anxious", "relieved", "frustrated"]
    assert pres.timeline[0].day_short == "Sen"
    assert pres.timeline[0].emotions[0].intensity == 8
    assert "anxious (8/10)" in pres.timeline[0].note
    assert pres.timeline[0].note.startswith("Kamu menulis tentang perasaan anxious (8/10). Kamu menyebut presentasi.")
    # entri multi-emosi: note menyebut keduanya
    assert "frustrated (8/10)" in pres.timeline[3].note
    assert "capek (5/10)" in pres.timeline[3].note
    assert len(pres.highlights) == 1
    assert pres.highlights[0].title == "Frustrated muncul di sekitar revisi"
    assert pres.highlights[0].supporting_entry_ids == s["candidate_patterns"][0]["entry_ids"]
    assert pres.reflection.question == "Q?"
    assert pres.explore.contexts == ["presentasi", "revisi", "kampus"]
    assert pres.explore.days == ["Sen", "Sel", "Rab", "Kam"]


def test_build_presentation_no_patterns_no_highlights():
    entries = [_e("2026-10-05", [{"emotion": "anxious", "intensity": 7}], "besok presentasi")]
    s = summarizer.build_structured(entries)
    pres = summarizer.build_presentation(s, "2026-10-05", "2026-10-11", "Q?")
    assert pres.highlights == []
    assert pres.overview_labels == ["anxious"]