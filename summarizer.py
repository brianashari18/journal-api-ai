"""Weekly emotional summary — HYBRID (multi-emosi per entri + intensitas user 1-10).

Bagian deterministik (auditable, tanpa LLM):
- entry_id stabil PER-ENTRI (uuid5 date|text) — bukan per emosi lagi.
- context (keyword), emotion_overview (label USER + frekuensi + intensitas mean/max/peak),
  timeline kronologis (semua emosi per entri, emosi utama = intensitas tertinggi),
  kandidat pattern (emosi+konteks, min 2 ENTRI), kandidat change (perubahan emosi utama).

Bagian naratif (LLM, ter-guardrail): deskripsi pattern/change via candidate_id -> server
mapping ke supporting_entry_ids; desc kausal/emosi di luar candidate -> template;
reflection nanya penyebab -> fallback. Presentation = layer user-facing deterministik.

Guarantee: supporting_entry_ids SELALU == entry_ids kandidat deterministik.
"""

import json
import uuid
from datetime import date
from typing import List, Optional

from ollama_client import chat_json
from schemas import (
    ChangeSummary,
    EmotionItem,
    EmotionOverview,
    Entry,
    ExploreBlock,
    Highlight,
    PatternSummary,
    Presentation,
    PresentationHeader,
    ReflectionBlock,
    TimelineItem,
    TimelineNote,
)

SYSTEM = (
    "Kamu adalah penulis ringkasan emosi mingguan untuk aplikasi jurnal. "
    "Gunakan bahasa observasional: 'you wrote...', 'in two entries...', 'it seems...'. "
    "JANGAN mengubah/mengganti label emosi user. JANGAN menebak emosi/intensitas baru. "
    "JANGAN mendiagnosa, JANGAN menyimpulkan sebab-akibat, JANGAN menebak penyebab. "
    "Deskripsi hanya boleh merujuk data candidate yang diberikan. "
    "Balas HANYA JSON murni, tanpa markdown."
)

_DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_DAYS_SHORT = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

# Tampilan Bahasa Indonesia untuk konteks (key internal tetap EN untuk deteksi).
_CONTEXT_DISPLAY = {
    "university": "kampus",
    "work": "kerjaan",
    "revision": "revisi",
    "deadline": "deadline",
    "project": "proyek",
    "presentation": "presentasi",
    "exercise": "olahraga",
    "friends": "teman",
    "family": "keluarga",
    "money": "uang",
}


def _ctx_display(ctx: str) -> str:
    return _CONTEXT_DISPLAY.get(ctx, ctx)

# Keyword -> konteks (EN + ID). Hanya konteks yang jelas tertulis di teks.
# Hati-hati substring overlap: "revisi" adalah substring "revisions" — jangan letakkan
# "revisi"/"proyek" di konteks lain selain revision/project biar gak dobel.
_CONTEXT_KEYWORDS = {
    "university": ["university", "lecturer", "class", "school", "kuliah", "dosen", "tugas"],
    "work": ["work", "meeting", "client", "kerjaan"],
    "revision": ["revision", "revisi"],
    "deadline": ["deadline", "tenggat"],
    "project": ["project", "proyek"],
    "presentation": ["presentation", "presentasi"],
    "exercise": ["exercise", "olahraga", "hiking", "gunung", "run"],
    "friends": ["friend", "teman", "lunch", "hang out", "nongkrong"],
    "family": ["family", "keluarga", "orang tua", "ibu", "ayah"],
    "money": ["salary", "gaji", "invest", "investasi", "uang"],
}


def _day_name(d: str) -> str:
    try:
        return _DAYS[date.fromisoformat(d).weekday()]
    except ValueError:
        return ""


def _day_short(d: str) -> str:
    try:
        return _DAYS_SHORT[date.fromisoformat(d).weekday()]
    except ValueError:
        return ""


def _period_label(start: str, end: str) -> str:
    def fmt(d: str) -> str:
        try:
            dt = date.fromisoformat(d)
            return f"{_MONTHS[dt.month - 1]} {dt.day}"
        except ValueError:
            return d

    return f"{fmt(start)} – {fmt(end)}"


def entry_id_for(entry: Entry) -> str:
    # PER-ENTRI: identitas = tanggal + teks (bukan per emosi lagi, 1 entri bisa 3 emosi).
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"journal|{entry.date}|{entry.text}"))


def extract_context(text: str) -> List[str]:
    lowered = text.lower()
    found = []
    for label, kws in _CONTEXT_KEYWORDS.items():
        if any(kw in lowered for kw in kws):
            found.append(label)
    return sorted(set(found))


def primary_emotion(emotions: List[EmotionItem]) -> str:
    """Emosi utama = intensitas tertinggi; seri -> yang pertama di list user."""
    if not emotions:
        return ""
    return max(emotions, key=lambda x: x.intensity if x.intensity else 0).emotion.strip()


def _emotion_label(item: EmotionItem) -> str:
    return f"{item.emotion} ({item.intensity}/10)" if item.intensity else item.emotion


def _template_pattern(emotion: str, context: str, n: int) -> str:
    return f"{emotion.capitalize()} muncul di {n} entri yang terkait {_ctx_display(context)}."


def _template_change(from_emotion: str, to_emotion: str) -> str:
    return f"Kamu menulis tentang {from_emotion}, lalu {to_emotion}."


def _note_for(t: dict) -> str:
    """Catatan observasional per entri (Bahasa Indonesia): semua emosi + intensitas + konteks."""
    parts = [_emotion_label(EmotionItem(**e)) for e in t.get("emotions", [])]
    base = "Kamu menulis tentang perasaan " + (" dan ".join(parts) if parts else "sesuatu") + "."
    if t.get("context"):
        base += " Kamu menyebut " + ", ".join(_ctx_display(c) for c in t["context"]) + "."
    return base


def build_structured(entries: List[Entry]) -> dict:
    """Bagian deterministik. Mengisi entry_id; mengembalikan overview/timeline (objek pydantic)
    + kandidat pattern/change (ber-candidate_id) + llm_input (tanpa quotes)."""
    for e in entries:
        if not e.entry_id:
            e.entry_id = entry_id_for(e)

    # Urut kronologis; tiebreak entry_id biar urutan same-day deterministik
    ordered = sorted(entries, key=lambda e: (e.date, e.entry_id))

    timeline: List[TimelineItem] = []
    for e in ordered:
        timeline.append(
            TimelineItem(
                date=e.date,
                day=_day_name(e.date) or e.day,
                emotion=primary_emotion(e.emotions),
                emotions=list(e.emotions),
                entry_id=e.entry_id,
            )
        )

    # emotion_overview per label USER + intensitas (mean/max/peak)
    groups: dict[str, dict] = {}
    for e in ordered:
        seen: set[str] = set()
        for item in e.emotions:
            label = item.emotion.strip()
            if not label or label in seen:
                continue
            seen.add(label)
            g = groups.setdefault(label, {"entry_ids": [], "quotes": [], "peaks": []})
            g["entry_ids"].append(e.entry_id)
            g["quotes"].append(e.text)
            if item.intensity:
                g["peaks"].append((item.intensity, e.entry_id, e.date))

    emotion_overview: List[EmotionOverview] = []
    for label, g in sorted(groups.items(), key=lambda kv: (-len(kv[1]["entry_ids"]), kv[0])):
        intensities = [p[0] for p in g["peaks"]]
        peak = max(g["peaks"], key=lambda p: p[0]) if g["peaks"] else None
        emotion_overview.append(
            EmotionOverview(
                emotion=label,
                entry_count=len(g["entry_ids"]),
                entry_ids=g["entry_ids"],
                quotes=g["quotes"],
                intensity_mean=round(sum(intensities) / len(intensities), 1) if intensities else None,
                intensity_max=max(intensities) if intensities else None,
                peak_entry_id=peak[1] if peak else "",
                peak_date=peak[2] if peak else "",
            )
        )

    # context per entry (untuk LLM & pattern)
    context_by_id = {e.entry_id: extract_context(e.text) for e in ordered}

    # kandidat pattern: pasangan (emosi, konteks) yang muncul di >= 2 ENTRI
    pair_map: dict[tuple[str, str], list[str]] = {}
    for e in ordered:
        seen: set[str] = set()
        for item in e.emotions:
            label = item.emotion.strip()
            if not label or label in seen:
                continue
            seen.add(label)
            for ctx in context_by_id[e.entry_id]:
                pair_map.setdefault((label, ctx), []).append(e.entry_id)
    candidate_patterns = [
        {
            "candidate_id": f"p{i}",
            "emotion": emo,
            "context": ctx,
            "entry_ids": sorted(ids),
        }
        for i, ((emo, ctx), ids) in enumerate(
            sorted(pair_map.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        )
        if len(ids) >= 2
    ]

    # kandidat change: perubahan EMOSI UTAMA antar entri kronologis berurutan
    candidate_changes = []
    for i, (prev, cur) in enumerate(zip(ordered, ordered[1:])):
        p = primary_emotion(prev.emotions)
        c = primary_emotion(cur.emotions)
        if p and c and p != c:
            candidate_changes.append(
                {
                    "candidate_id": f"c{i}",
                    "from_emotion": p,
                    "to_emotion": c,
                    "from_date": prev.date,
                    "to_date": cur.date,
                    "entry_ids": [prev.entry_id, cur.entry_id],
                }
            )

    llm_input = {
        "emotion_overview": [
            {"emotion": o.emotion, "entry_count": o.entry_count, "entry_ids": o.entry_ids,
             "intensity_mean": o.intensity_mean, "intensity_max": o.intensity_max}
            for o in emotion_overview
        ],
        "timeline": [
            {"date": t.date, "day": t.day, "emotion": t.emotion, "entry_id": t.entry_id,
             "context": context_by_id.get(t.entry_id, []),
             "emotions": [e.model_dump() for e in t.emotions]}
            for t in timeline
        ],
        "candidate_patterns": candidate_patterns,
        "candidate_changes": candidate_changes,
    }

    return {
        "timeline": timeline,
        "emotion_overview": emotion_overview,
        "candidate_patterns": candidate_patterns,
        "candidate_changes": candidate_changes,
        "llm_input": llm_input,
    }


def build_narrative_prompt(llm_input: dict) -> str:
    return (
        "Berdasarkan data terstruktur berikut (source of truth):\n\n"
        "EMOTION OVERVIEW:\n" + json.dumps(llm_input["emotion_overview"], ensure_ascii=False) +
        "\n\nTIMELINE (date|day|primary_emotion|context|all_emotions):\n" +
        "\n".join(
            f"{t['date']}|{t['day']}|{t['emotion']}|{','.join(t['context'])}|{t['emotions']}"
            for t in llm_input["timeline"]
        ) +
        "\n\nCANDIDATE PATTERNS (setiap item SUDAH lolos bukti min 2 entri):\n" +
        json.dumps(llm_input["candidate_patterns"], ensure_ascii=False) +
        "\n\nCANDIDATE CHANGES (perubahan emosi utama kronologis):\n" +
        json.dumps(llm_input["candidate_changes"], ensure_ascii=False) +
        "\n\nTulis hasil untuk SETIAP candidate (satu per satu, wajib pakai candidate_id persis):\n"
        '- "patterns": satu objek {"candidate_id": str, "description": str} per candidate pattern; '
        "deskripsi observasional singkat, hanya boleh merujuk hal yang ada di candidate itu.\n"
        '- "changes": satu objek {"candidate_id": str, "description": str} per candidate change; '
        "deskripsi observasional singkat (mis. 'You wrote about feeling {from} on {from_date}, "
        "then {to} on {to_date}.'), hanya merujuk candidate itu. "
        "Setiap deskripsi HANYA boleh menyebut emosi & tanggal dari candidate-nya — "
        "jangan sebut emosi/tanggal/konteks dari entri lain.\n"
        '- "reflection_question": SATU pertanyaan terbuka non-directive yang mengundang user '
        "menafsirkan pengalamannya sendiri, berdasar salah satu pattern/change. "
        "JANGAN menanyakan penyebab ('kenapa', 'apa yang membuat', 'apa penyebab') — "
        "tanyakan hal yang bisa user perhatikan sendiri (mis. 'apa yang membantu kamu...'). "
        "Kalau tidak ada pattern maupun change, isi string kosong.\n"
        "Semua deskripsi ditulis dalam Bahasa Indonesia; JANGAN menerjemahkan/mengubah "
        "label emosi atau konteks — tetap persis seperti di data.\n"
        "DILARANG: membuat pattern/change baru di luar candidate, mengganti label emosi, "
        "menebak penyebab, diagnosa.\n"
        'Balas HANYA JSON: '
        '{"patterns":[{"candidate_id":str,"description":str}],'
        '"changes":[{"candidate_id":str,"description":str}],'
        '"reflection_question":str}'
    )


def _parse_narrative(raw: str) -> tuple[List[dict], List[dict], str]:
    data = json.loads(raw)
    patterns = [
        {"candidate_id": str(p.get("candidate_id", "")), "description": str(p.get("description", "")).strip()}
        for p in data.get("patterns", []) if isinstance(p, dict)
    ]
    changes = [
        {"candidate_id": str(c.get("candidate_id", "")), "description": str(c.get("description", "")).strip()}
        for c in data.get("changes", []) if isinstance(c, dict)
    ]
    return patterns, changes, str(data.get("reflection_question", "")).strip()


_CAUSAL_WORDS = ["karena", "because", "caused", "penyebab", "sehingga", "mengakibatkan", "akibat"]

# Reflection question dilarang nanya penyebab — kalau kena, fallback ke pertanyaan generik aman.
_REFLECTION_BANNED = [
    "karena", "because", "caused", "penyebab", "sehingga", "mengakibatkan",
    "membuat", "kenapa", "apa yang membuat", "apa penyebab", "yang menyebabkan",
]
_REFLECTION_FALLBACK = "Apa yang kamu perhatikan dari perasaanmu minggu ini?"


def _desc_ok(desc: str, required: List[str]) -> bool:
    """Deskripsi LLM lolos kalau: menyebut semua token wajib candidate & tanpa kata kausal."""
    lowered = desc.lower()
    if any(w in lowered for w in _CAUSAL_WORDS):
        return False
    return all(req in lowered for req in required)


def _reflection_ok(q: str) -> bool:
    return q != "" and not any(w in q.lower() for w in _REFLECTION_BANNED)


def _assemble(candidates_p: list, candidates_c: list, parsed_p: list, parsed_c: list,
              reflection: str) -> tuple[List[PatternSummary], List[ChangeSummary], str]:
    """Gabungkan deskripsi LLM dengan evidence deterministik (candidate_id -> entry_ids).

    Deskripsi LLM yang melanggar (kata kausal / nyebut emosi di luar candidate) -> template.
    ID dari LLM yang tidak dikenal -> dibuang. Guarantee: supporting_entry_ids == entry_ids
    kandidat, dan deskripsi tidak pernah menyimpulkan sebab-akibat.
    """
    p_desc = {x["candidate_id"]: x["description"] for x in parsed_p if x["candidate_id"]}
    c_desc = {x["candidate_id"]: x["description"] for x in parsed_c if x["candidate_id"]}

    patterns = []
    for c in candidates_p:
        desc = p_desc.get(c["candidate_id"], "")
        if not _desc_ok(desc, [c["emotion"], c["context"]]):
            desc = _template_pattern(c["emotion"], c["context"], len(c["entry_ids"]))
        patterns.append(PatternSummary(description=desc, supporting_entry_ids=c["entry_ids"]))

    changes = []
    for c in candidates_c:
        desc = c_desc.get(c["candidate_id"], "")
        if not _desc_ok(desc, [c["from_emotion"], c["to_emotion"]]):
            desc = _template_change(c["from_emotion"], c["to_emotion"])
        changes.append(ChangeSummary(description=desc, supporting_entry_ids=c["entry_ids"]))

    if not _reflection_ok(reflection):
        reflection = _REFLECTION_FALLBACK

    return patterns, changes, reflection


def generate_narrative(structured: dict) -> tuple[List[PatternSummary], List[ChangeSummary], str]:
    """LLM menulis deskripsi per kandidat. Retry 1x. Evidence selalu dari kandidat deterministik."""
    candidates_p = structured["candidate_patterns"]
    candidates_c = structured["candidate_changes"]
    prompt = build_narrative_prompt(structured["llm_input"])
    for attempt in range(2):
        raw = chat_json(SYSTEM, prompt)
        try:
            parsed_p, parsed_c, reflection = _parse_narrative(raw)
            return _assemble(candidates_p, candidates_c, parsed_p, parsed_c, reflection)
        except (json.JSONDecodeError, TypeError, ValueError):
            print(f"[summarizer] narrative JSON parse gagal (attempt {attempt + 1}), raw:\n{raw}")
            if attempt == 0:
                prompt += "\n\nPENTING: balas hanya JSON murni."
    print("[summarizer] fallback ke template (LLM gagal) — evidence tetap terjaga")
    return _assemble(candidates_p, candidates_c, [], [], "")


def build_presentation(structured: dict, start: str, end: str, reflection_question: str) -> Presentation:
    """Layer user-facing (deterministik): header, chips, intro, timeline notes, highlights,
    disclaimer, reflection, explore. Semua dari evidence yang sama — tanpa penambahan LLM."""
    tlines = structured["llm_input"]["timeline"]  # [{date,day,emotion,entry_id,context,emotions}]

    # urutan kemunculan pertama (untuk chips + explore)
    ordered_emotions: List[str] = []
    for t in tlines:
        emo = t["emotion"].strip()
        if emo and emo not in ordered_emotions:
            ordered_emotions.append(emo)

    contexts_ordered: List[str] = []
    for t in tlines:
        for ctx in t["context"]:
            if ctx not in contexts_ordered:
                contexts_ordered.append(ctx)

    notes = [
        TimelineNote(
            day_short=_day_short(t["date"]),
            day=t["day"],
            emotion=t["emotion"],
            emotions=[EmotionItem(**e) for e in t.get("emotions", [])],
            note=_note_for(t),
            entry_id=t["entry_id"],
        )
        for t in tlines
    ]

    highlights = []
    for c in structured["candidate_patterns"]:
        n = len(c["entry_ids"])
        highlights.append(
            Highlight(
                title=f"{c['emotion'].capitalize()} muncul di sekitar {_ctx_display(c['context'])}",
                body=(
                    f"Kamu menyebut {c['emotion']} dalam {n} entri, "
                    f"{'keduanya terkait' if n == 2 else 'semuanya terkait'} {_ctx_display(c['context'])}."
                ),
                supporting_entry_ids=c["entry_ids"],
            )
        )

    # de-dupe urutan hari (biar "Jelajahi minggu ini" bersih)
    days_ordered: List[str] = []
    for t in tlines:
        d = _day_short(t["date"])
        if d and d not in days_ordered:
            days_ordered.append(d)

    return Presentation(
        header=PresentationHeader(title="Minggu Emosimu", period_label=_period_label(start, end)),
        overview_labels=ordered_emotions,
        intro=(
            f"Minggu ini kamu melalui {len(ordered_emotions)} emosi berbeda: "
            f"{', '.join(ordered_emotions)}."
        ),
        timeline=notes,
        highlights=highlights,
        disclaimer=(
            "Ini adalah pengamatan dari yang kamu tulis minggu ini, bukan kesimpulan "
            "tentang mengapa kamu merasa dengan cara tertentu."
        ),
        reflection=ReflectionBlock(
            prompt=(
                "Melihat kembali pengalaman-pengalaman ini, apa yang kamu perhatikan "
                "saat perasaanmu berubah?"
            ),
            question=reflection_question,
        ),
        explore=ExploreBlock(
            emotions=ordered_emotions,
            contexts=[_ctx_display(c) for c in contexts_ordered],
            days=days_ordered,
        ),
    )