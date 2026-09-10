from typing import List, Optional

from pydantic import BaseModel


class EmotionItem(BaseModel):
    emotion: str
    intensity: Optional[int] = None  # 1-10, diisi USER (bukan tebakan mesin)


class Entry(BaseModel):
    date: str
    day: str = ""
    text: str
    emotions: List[EmotionItem] = []  # maks 3, pilihan user
    entry_id: str = ""


class SummarizeRequest(BaseModel):
    start_date: str
    end_date: str
    entries: List[Entry] = []


class PeriodSummary(BaseModel):
    start: str
    end: str
    entry_count: int


class EmotionOverview(BaseModel):
    emotion: str
    entry_count: int
    entry_ids: List[str] = []
    quotes: List[str] = []  # kutipan verbatim (cermin)
    intensity_mean: Optional[float] = None
    intensity_max: Optional[int] = None
    peak_entry_id: str = ""
    peak_date: str = ""


class TimelineItem(BaseModel):
    date: str
    day: str = ""
    emotion: str  # emosi utama (intensitas tertinggi) utk traceability/change
    emotions: List[EmotionItem] = []  # semua emosi entri (maks 3)
    entry_id: str = ""


class PatternSummary(BaseModel):
    description: str
    supporting_entry_ids: List[str] = []


class ChangeSummary(BaseModel):
    description: str
    supporting_entry_ids: List[str] = []


class WeeklySummaryResponse(BaseModel):
    period: PeriodSummary
    emotion_overview: List[EmotionOverview]
    timeline: List[TimelineItem]
    patterns: List[PatternSummary] = []
    changes: List[ChangeSummary] = []
    reflection_question: str = ""
    presentation: Optional["Presentation"] = None


class PresentationHeader(BaseModel):
    title: str
    period_label: str


class TimelineNote(BaseModel):
    day_short: str
    day: str
    emotion: str  # emosi utama, utk fallback/header sederhana
    emotions: List[EmotionItem] = []  # semua emosi entri (maks 3) + intensitas
    note: str
    entry_id: str


class Highlight(BaseModel):
    title: str
    body: str
    supporting_entry_ids: List[str] = []


class ReflectionBlock(BaseModel):
    prompt: str
    question: str


class ExploreBlock(BaseModel):
    emotions: List[str] = []
    contexts: List[str] = []
    days: List[str] = []


class Presentation(BaseModel):
    header: PresentationHeader
    overview_labels: List[str] = []
    intro: str
    timeline: List[TimelineNote] = []
    highlights: List[Highlight] = []
    disclaimer: str
    reflection: ReflectionBlock
    explore: ExploreBlock


WeeklySummaryResponse.model_rebuild()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]