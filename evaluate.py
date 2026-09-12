"""Evaluasi fidelity summary emosi mingguan (HYBRID, multi-emosi + intensitas).

Cek bagian DETERMINISTIK yang bisa diverifikasi tanpa LLM:
- emotion_overview: label user == label di summary, count & entry_ids sesuai,
  quotes verbatim, intensitas (mean) sesuai input user, peak ada kalau data intensitas ada.

Narasi LLM (pattern/change/reflection) TIDAK dievaluasi di sini — itu subjektif & perlu
subjek/user. Yang dipastikan: evidence (label user, entry_id, kutipan, intensitas) setia.

Usage: python evaluate.py [start_date] [end_date]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.repository import qdrant_store  # noqa: E402
from app.services import summarizer  # noqa: E402
from app.schemas.journal import EmotionItem, Entry  # noqa: E402

DATA = Path(__file__).parent / "data" / "journals" / "sample.json"


def load_entries() -> list[Entry]:
    raw = json.loads(DATA.read_text())
    out = []
    for e in raw:
        out.append(
            Entry(
                date=e["date"],
                day="",
                text=e["text"],
                emotions=[EmotionItem(**x) for x in e.get("emotions", [])],
            )
        )
    return out


def labels_of(entries: list[Entry]) -> set[str]:
    return {x.emotion.strip() for e in entries for x in e.emotions if x.emotion.strip()}


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "2026-08-31"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-09-06"

    entries = load_entries()
    in_range = [e for e in entries if start <= e.date <= end]
    print(f"Evaluasi cermin {start} s/d {end}: {len(in_range)} entri (multi-emosi + intensitas)")

    qdrant_store.ensure_collection()
    qdrant_store.upsert_entries(entries)
    retrieved = qdrant_store.retrieve_by_date(start, end)
    print(f"Retrieved dari Qdrant (filter tanggal): {len(retrieved)} entri")

    structured = summarizer.build_structured(retrieved)
    by_label = {o.emotion: o for o in structured["emotion_overview"]}

    user_labels = sorted(labels_of(in_range))
    summary_labels = sorted(by_label)
    print(f"Label user: {user_labels}")
    print(f"Label di summary: {summary_labels}")

    print("\nPer-emosi (fidelity cermin, bagian deterministik):")
    all_ok = True
    for label in user_labels:
        o = by_label.get(label)
        if o is None:
            print(f"  {label:<12} MISS (label user hilang)")
            all_ok = False
            continue
        expected = [
            (e.entry_id if e.entry_id else summarizer.entry_id_for(e), e.text,
             [x.intensity for x in e.emotions if x.emotion.strip() == label and x.intensity])
            for e in in_range if any(x.emotion.strip() == label for x in e.emotions)
        ]
        count_ok = o.entry_count == len(expected)
        ids_ok = len(o.entry_ids) == len(expected) and all(o.entry_ids)
        quotes_ok = len(o.quotes) == len(expected) and all(
            o.quotes[i] == expected[i][1] for i in range(len(expected))
        )
        intensities = [iv for (_, _, ivs) in expected for iv in ivs]
        if intensities:
            mean_ok = o.intensity_mean is not None and abs(o.intensity_mean - round(sum(intensities) / len(intensities), 1)) < 0.05
            peak_ok = o.intensity_max == max(intensities) and o.peak_entry_id and o.peak_date
        else:
            mean_ok = o.intensity_mean is None
            peak_ok = True
        status = "OK" if (count_ok and ids_ok and quotes_ok and mean_ok and peak_ok) else "GAGAL"
        if status != "OK":
            all_ok = False
        print(
            f"  {label:<12} count={o.entry_count}({len(expected)}) ids={ids_ok} "
            f"quotes={quotes_ok} mean={mean_ok} peak={peak_ok} -> {status}"
        )

    print(f"\nKandidat pattern (min 2 entri): {len(structured['candidate_patterns'])}")
    print(f"Kandidat change (emosi utama): {len(structured['candidate_changes'])}")
    print(f"\nVERDICT: {'cermin SETIA (label + entry_id + kutipan + intensitas)' if all_ok else 'ADA GAGAL'}")


if __name__ == "__main__":
    main()