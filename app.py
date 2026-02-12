import os
import re
import json
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
import requests
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

# ========= App Config =========
APP_TITLE = "Italian Vocab from YouTube — LLM-Only (Multi-source Captions)"
MAX_CONTEXT_CHARS = 12000    # cap transcript length sent to LLM
DEFAULT_MAX_CARDS = 18

# ========= Utility: YouTube ID parsing =========

def extract_video_id(url_or_id: str) -> Optional[str]:
    """
    Extract an 11-char YouTube video ID from a URL, or accept a bare ID.
    Handles: watch URLs, youtu.be, Shorts, typical query params.
    """
    if not url_or_id:
        return None
    url_or_id = url_or_id.strip()
    # Bare ID
    if re.match(r"^[\w-]{11}$", url_or_id):
        return url_or_id
    try:
        parsed = urlparse(url_or_id)
        q = parse_qs(parsed.query)
        # youtu.be/<id>
        if parsed.netloc in ("youtu.be", "www.youtu.be"):
            vid = parsed.path.lstrip("/")
            return vid if re.match(r"^[\w-]{11}$", vid) else None
        # youtube.com/watch?v=<id>
        if parsed.netloc in ("youtube.com", "www.youtube.com", "m.youtube.com"):
            if "v" in q:
                v = q.get("v", [None])[0]
                return v if v and re.match(r"^[\w-]{11}$", v) else None
            # /shorts/<id>
            if parsed.path.startswith("/shorts/"):
                parts = [p for p in parsed.path.split("/") if p]
                vid = parts[1] if len(parts) > 1 else None
                return vid if vid and re.match(r"^[\w-]{11}$", vid) else None
    except Exception:
        return None
    return None

# ========= Primary captions: youtube-transcript-api =========

@st.cache_data(show_spinner=False)
def list_available_transcripts(video_id: str):
    """Return TranscriptList or None if inaccessible."""
    try:
        return YouTubeTranscriptApi.list_transcripts(video_id)
    except (TranscriptsDisabled, CouldNotRetrieveTranscript, Exception):
        return None

@st.cache_data(show_spinner=False)
def fetch_transcript_ytta(
    video_id: str,
    preferred_langs=('it','it-IT','en','en-US','en-GB'),
    allow_any_fallback: bool = True,
) -> Optional[str]:
    """
    Primary method: enumerate manual + auto transcripts and fetch one.
    """
    transcripts = list_available_transcripts(video_id)
    if transcripts is None:
        return None

    # Manual in preferred languages
    for lang in preferred_langs:
        try:
            t = transcripts.find_transcript([lang])
            entries = t.fetch()
            return " ".join([e["text"] for e in entries])
        except Exception:
            pass

    # Auto-generated in preferred languages
    for lang in preferred_langs:
        try:
            t = transcripts.find_generated_transcript([lang])
            entries = t.fetch()
            return " ".join([e["text"] for e in entries])
        except Exception:
            pass

    if allow_any_fallback:
        # Any manual
        try:
            for t in transcripts.transcripts:
                if not t.is_generated:
                    entries = t.fetch()
                    return " ".join([e["text"] for e in entries])
        except Exception:
            pass
        # Any auto
        try:
            for t in transcripts.transcripts:
                if t.is_generated:
                    entries = t.fetch()
                    return " ".join([e["text"] for e in entries])
        except Exception:
            pass

    return None

# ========= Fallback A: TranscriptAPI.com =========
# https://transcriptapi.com/docs/api/
def fetch_transcript_transcriptapi(video_url_or_id: str, language_priority=("it","en")) -> Optional[str]:
    """
    Uses TranscriptAPI.com if TRANSCRIPTAPI_API_KEY is provided in secrets.
    Returns transcript text or None on failure.
    """
    api_key = st.secrets.get("TRANSCRIPTAPI_API_KEY", None)
    if not api_key:
        return None
    # Accept either full URL or ID
    vid = extract_video_id(video_url_or_id) or video_url_or_id.strip()
    # The API accepts either full URL or ID; use the ID to be safe
    url = "https://transcriptapi.com/api/v2/youtube/transcript"
    headers = {"Authorization": f"Bearer {api_key}"}
    # Try preferred languages in order; if none, get default
    for lang in list(language_priority) + [None]:
        params = {"video_url": vid}
        if lang:
            params["lang"] = lang
        try:
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if "transcript" in data and isinstance(data["transcript"], list):
                    return " ".join([seg.get("text","") for seg in data["transcript"]])
        except Exception:
            pass
    return None

# ========= Fallback B: YouTubeTranscript.dev =========
# https://github.com/volodstaimi/Youtube-Transcript-API
def fetch_transcript_ytdtranscriptdev(video_url_or_id: str) -> Optional[str]:
    """
    Uses youtubetranscript.dev if YTDEV_API_KEY is provided in secrets.
    Returns transcript text or None on failure.
    """
    api_key = st.secrets.get("YTDEV_API_KEY", None)
    if not api_key:
        return None
    vid = extract_video_id(video_url_or_id) or video_url_or_id.strip()
    url = "https://youtubetranscript.dev/api/v2/transcribe"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"video": vid}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            # V2 may return different shapes; handle common "paragraphs" or "transcript"
            if isinstance(data, dict):
                if "transcript" in data and isinstance(data["transcript"], list):
                    return " ".join([seg.get("text","") for seg in data["transcript"]])
                if "paragraphs" in data and isinstance(data["paragraphs"], list):
                    return " ".join([p.get("text","") for p in data["paragraphs"]])
    except Exception:
        pass
    return None

# ========= Manual upload & parsing =========

def parse_srt(content: str) -> str:
    # Remove index lines and timecodes like "00:00:01,000 --> 00:00:03,000"
    lines = []
    for line in content.splitlines():
        line = line.strip("\ufeff ").strip()
        if re.match(r"^\d+$", line):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}$", line):
            continue
        if line and not line.startswith("{\\an"):  # skip some styling tags
            lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()

def parse_vtt(content: str) -> str:
    content = content.replace("\ufeff","").strip()
    content = re.sub(r"^WEBVTT.*?\n", "", content, flags=re.IGNORECASE|re.DOTALL)
    # Remove timestamps like 00:00.000 --> 00:01.500
    text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}:\d{2}\.\d{3}.*", "", content)
    text = re.sub(r"\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}\.\d{3}.*", "", text)
    text = re.sub(r"<[^>]+>", "", text)  # strip tags
    return re.sub(r"\s+", " ", text).strip()

def normalize(text: str) -> str:
    text = text.replace("\u200b", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text

# ========= LLM prompt & call =========

def build_flashcard_prompt(transcript: str, max_items: int, level: str) -> str:
    transcript = transcript[:MAX_CONTEXT_CHARS]
    return f"""
You are an expert Italian language tutor.

Task:
1) Read the Italian transcript below.
2) Choose up to {max_items} vocabulary ITEMS (single words or multi‑word expressions) that are valuable to study for an {level} learner.
3) For each item, generate a high‑quality flashcard.

Selection criteria:
- Only include items that APPEAR in the transcript.
- Prefer items that are meaningful for understanding, moderately challenging, reusable, and thematically relevant.
- Include a mix of verbs, nouns, adjectives/adverbs, and idiomatic/multi‑word expressions when appropriate.
- Avoid articles, simple pronouns, very basic function words, and extremely rare proper names.

Output format (STRICT JSON ONLY, no markdown, no extra text):
[
  {{
    "term_it": "…",
    "pos": "…",
    "translation_en": "…",
    "example_it": "…",
    "gloss_en": "…",
    "rationale": "…"
  }},
  ...
]

Transcript:
{transcript}
"""

def _try_parse_llm_json(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("["); end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try: return json.loads(raw[start:end+1])
        except Exception: pass
    return None

def call_llm_for_flashcards(transcript: str, max_items: int, level: str, temperature: float = 0.3) -> List[Dict]:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit Secrets. Add it via: ⋮ → Edit secrets.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    prompt = build_flashcard_prompt(transcript, max_items, level)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
    )
    raw = resp.choices[0].message.content.strip()

    data = _try_parse_llm_json(raw)
    if data is None or not isinstance(data, list):
        raise ValueError("Could not parse LLM JSON.\n\nRaw (first 800 chars):\n" + raw[:800])

    required = {"term_it","pos","translation_en","example_it","gloss_en","rationale"}
    cleaned = []
    for item in data:
        if not isinstance(item, dict): continue
        if not required.issubset(item.keys()): continue
        for k,v in list(item.items()):
            if isinstance(v, str): item[k] = v.strip()
        cleaned.append(item)
    if not cleaned:
        raise ValueError("LLM returned no usable items after validation.")
    return cleaned

def flashcards_to_dataframe(cards: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(cards)
    cols = ["term_it","pos","translation_en","example_it","gloss_en","rationale"]
    return df[cols]

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    export = pd.DataFrame({
        "Front": df["term_it"],
        "Back": df.apply(lambda r: f"({r['pos']}) — {r['translation_en']}<br><br>"
                                   f"IT: {r['example_it']}<br>"
                                   f"EN: {r['gloss_en']}", axis=1)
    })
    return export.to_csv(index=False).encode("utf-8")

# ========= Orchestration =========

def get_transcript_text(
    youtube_input: str,
    manual_text: str,
    uploaded_file,
    preferred_langs: tuple,
    allow_any_fallback: bool
) -> Optional[str]:
    """
    Returns normalized transcript text from one of:
    - pasted text
    - uploaded .txt/.srt/.vtt
    - youtube-transcript-api (primary)
    - transcriptapi.com (fallback A) if key present
    - youtubetranscript.dev (fallback B) if key present
    """
    # A) Manual text takes precedence
    if manual_text and manual_text.strip():
        return normalize(manual_text)

    # B) File upload
    if uploaded_file is not None:
        raw = uploaded_file.read().decode("utf-8", errors="ignore")
        name = uploaded_file.name.lower()
        if name.endswith(".srt"):
            return normalize(parse_srt(raw))
        if name.endswith(".vtt"):
            return normalize(parse_vtt(raw))
        # assume plain text
        return normalize(raw)

    # C) YouTube link/ID path
    vid = extract_video_id(youtube_input)
    if not vid:
        return None

    # Primary
    text = fetch_transcript_ytta(vid, preferred_langs, allow_any_fallback)
    if text: return normalize(text)

    # Fallback A: TranscriptAPI.com
    text = fetch_transcript_transcriptapi(youtube_input)
    if text: return normalize(text)

    # Fallback B: YouTubeTranscript.dev
    text = fetch_transcript_ytdtranscriptdev(youtube_input)
    if text: return normalize(text)

    return None

# ========= UI =========

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown(
    "Paste a **YouTube** link (or provide a transcript). The app fetches captions when available, "
    "falls back to optional transcript APIs (if keys are configured), or uses pasted/uploaded captions—"
    "then asks an LLM to generate **Italian vocabulary flashcards**."
)

# Sidebar controls
st.sidebar.header("Settings")
max_cards = st.sidebar.slider("Number of flashcards", 6, 30, DEFAULT_MAX_CARDS, step=2)
level = st.sidebar.selectbox("Learner level focus", ["A2", "B1", "B2"])
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)
preferred_langs = st.sidebar.multiselect(
    "Preferred transcript languages (ordered)",
    options=["it","it-IT","en","en-US","en-GB","es","fr","de"],
    default=["it","it-IT","en","en-US","en-GB"],
)
allow_any_fallback = st.sidebar.checkbox("If preferred languages missing, allow any available transcript", value=True)
show_rationales = st.sidebar.checkbox("Show rationale (why each item was chosen)", value=False)

st.sidebar.markdown("**Secrets supported:** `OPENAI_API_KEY`, `OPENAI_MODEL` (optional), "
                    "`TRANSCRIPTAPI_API_KEY` (optional), `YTDEV_API_KEY` (optional).")

# Inputs
youtube_url = st.text_input("YouTube URL (or 11-character ID)")
manual_text = st.text_area("Or paste the Italian transcript here (overrides URL if filled)", height=160)
uploaded = st.file_uploader("…or upload a transcript file (.txt, .srt, .vtt)", type=["txt","srt","vtt"])

# Debug panel (optional): show what ytta can see for this video
if youtube_url.strip():
    vid_dbg = extract_video_id(youtube_url)
    if vid_dbg:
        with st.expander("DEBUG: Available transcript tracks via youtube-transcript-api"):
            ts = list_available_transcripts(vid_dbg)
            if ts is None:
                st.write("No transcript list available (captions disabled/restricted/inaccessible from this host).")
            else:
                rows = [{"language_code": t.language_code, "is_generated": t.is_generated} for t in ts.transcripts]
                st.dataframe(pd.DataFrame(rows))

if st.button("Generate flashcards"):
    transcript = get_transcript_text(
        youtube_input=youtube_url,
        manual_text=manual_text,
        uploaded_file=uploaded,
        preferred_langs=tuple(preferred_langs),
        allow_any_fallback=allow_any_fallback
    )
    if not transcript:
        st.warning(
            "Could not obtain a transcript. Try pasting the transcript text, uploading an .srt/.vtt/.txt file, "
            "or add a fallback API key in Secrets."
        )
        st.stop()

    with st.spinner("Asking the LLM to curate vocabulary and build flashcards…"):
        try:
            cards = call_llm_for_flashcards(transcript, max_cards, level, temperature=temperature)
        except Exception as e:
            st.error(str(e))
            st.stop()

    df = flashcards_to_dataframe(cards)

    st.subheader("Flashcards")
    for _, r in df.iterrows():
        st.markdown(
            f"**{r['term_it']}** ({r['pos']}) — {r['translation_en']}\n\n"
            f"**IT:** {r['example_it']}\n\n"
            f"**EN:** {r['gloss_en']}\n"
        )
        if show_rationales:
            st.caption(f"Why this item: {r['rationale']}")

    st.subheader("Table view")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="⬇️ Download CSV for Anki (Front=term_it, Back=definition+examples)",
        data=df_to_csv_bytes(df),
        file_name="italian_flashcards.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Primary captions use the open-source `youtube-transcript-api` (manual/auto captions; no YouTube API key). "
    "If blocked, configure a fallback API in Secrets, or paste/upload transcripts instead."
)
