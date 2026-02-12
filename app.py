import os
import re
import json
from urllib.parse import urlparse, parse_qs
from typing import List, Dict

import streamlit as st
import pandas as pd
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

# ========= App Config =========
APP_TITLE = "Italian Vocab from YouTube — LLM-Only (Robust Captions)"
MAX_CONTEXT_CHARS = 12000    # cap transcript length sent to LLM
DEFAULT_MAX_CARDS = 18

# ========= Helpers =========

def extract_video_id(url_or_id: str) -> str | None:
    """
    Extract an 11-char YouTube video ID from a URL, or accept a bare ID.
    Handles: full watch URLs, youtu.be, Shorts, and common query params.
    """
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

        # youtube.com/watch?v=<id> (ignore playlists/timestamps etc.)
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


@st.cache_data(show_spinner=False)
def list_available_transcripts(video_id: str):
    """
    Return the TranscriptList for a video (or None if not accessible).
    This is separated so we can show it in a debug panel without refetching.
    """
    try:
        return YouTubeTranscriptApi.list_transcripts(video_id)
    except (TranscriptsDisabled, CouldNotRetrieveTranscript, Exception):
        return None


@st.cache_data(show_spinner=False)
def fetch_transcript_text(
    video_id: str,
    preferred_langs=('it','it-IT','en','en-US','en-GB'),
    allow_any_fallback: bool = True,
) -> str | None:
    """
    Robust transcript retrieval:
    1) List all available transcripts (manual + auto).
    2) Try preferred languages in order (manual first, then auto).
    3) Fallback to ANY available transcript (manual first) if allowed.
    Returns a single string (joined text) or None if nothing is accessible.
    """
    transcripts = list_available_transcripts(video_id)
    if transcripts is None:
        return None

    # Try manual transcripts in preferred languages
    for lang in preferred_langs:
        try:
            t = transcripts.find_transcript([lang])
            entries = t.fetch()
            return " ".join([e["text"] for e in entries])
        except Exception:
            pass

    # Try auto-generated transcripts in preferred languages
    for lang in preferred_langs:
        try:
            t = transcripts.find_generated_transcript([lang])
            entries = t.fetch()
            return " ".join([e["text"] for e in entries])
        except Exception:
            pass

    if allow_any_fallback:
        # Fallback: pick ANY manual transcript
        try:
            for t in transcripts.transcripts:
                if not t.is_generated:
                    entries = t.fetch()
                    return " ".join([e["text"] for e in entries])
        except Exception:
            pass

        # Fallback: pick ANY auto transcript
        try:
            for t in transcripts.transcripts:
                if t.is_generated:
                    entries = t.fetch()
                    return " ".join([e["text"] for e in entries])
        except Exception:
            pass

    return None


def normalize(text: str) -> str:
    # Basic cleanup—keep accents, normalize whitespace
    text = text.replace("\u200b", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_flashcard_prompt(transcript: str, max_items: int, level: str) -> str:
    """
    Ask the LLM to select the most pedagogically useful vocabulary
    and return STRICT JSON for easy parsing and CSV export.
    """
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
    """Try strict JSON; if it fails, grab the first [...] block heuristically."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Heuristic: find first '[' and last ']' and try again
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end+1])
        except Exception:
            pass
    # Give up; return None so caller can raise a helpful error
    return None


def call_llm_for_flashcards(transcript: str, max_items: int, level: str, temperature: float = 0.3) -> List[Dict]:
    """
    Uses an OpenAI-compatible Chat Completions API (OPENAI_API_KEY in secrets).
    Returns list[dict] with the required flashcard fields.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit Secrets. Add it via: ⋮ → Edit secrets.")

    from openai import OpenAI  # lazy import
    client = OpenAI(api_key=api_key)

    prompt = build_flashcard_prompt(transcript, max_items, level)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    raw = resp.choices[0].message.content.strip()

    data = _try_parse_llm_json(raw)
    if data is None or not isinstance(data, list):
        raise ValueError(
            "Could not parse LLM JSON.\n\n"
            f"Raw output (first 800 chars):\n{raw[:800]}"
        )

    # Basic schema validation
    required = {"term_it", "pos", "translation_en", "example_it", "gloss_en", "rationale"}
    cleaned = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        missing = required - set(item.keys())
        if missing:
            continue
        # Strip whitespace
        for k in list(item.keys()):
            if isinstance(item[k], str):
                item[k] = item[k].strip()
        cleaned.append(item)

    if not cleaned:
        raise ValueError("LLM returned no usable items after validation.")
    return cleaned


def flashcards_to_dataframe(cards: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(cards)
    cols = ["term_it", "pos", "translation_en", "example_it", "gloss_en", "rationale"]
    return df[cols]


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Export a Front/Back CSV suitable for Anki (UTF‑8).
    Back contains POS + translation + example + gloss (with <br> line breaks).
    """
    export_basic = pd.DataFrame({
        "Front": df["term_it"],
        "Back": df.apply(
            lambda r: f"({r['pos']}) — {r['translation_en']}<br><br>"
                      f"IT: {r['example_it']}<br>"
                      f"EN: {r['gloss_en']}",
            axis=1
        )
    })
    return export_basic.to_csv(index=False).encode("utf-8")


# ========= UI =========

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.markdown(
    "Paste a **YouTube** link (or a transcript). The app fetches captions when available, "
    "sends the text to an LLM, and returns curated **Italian vocabulary flashcards**—no frequency list needed."
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
debug_transcripts = st.sidebar.checkbox("Debug: show available transcripts for the video", value=False)

st.sidebar.markdown(
    "**LLM key:** Add `OPENAI_API_KEY` (and optional `OPENAI_MODEL`) via app menu **⋮ → Edit secrets**."
)

# Inputs
youtube_url = st.text_input("YouTube URL (or 11-character video ID)")
manual_text = st.text_area("Or paste the Italian transcript here (overrides the URL if filled)", height=180)

if st.button("Generate flashcards"):
    # 1) Transcript
    if manual_text.strip():
        raw_text = manual_text.strip()
        vid = None
    else:
        vid = extract_video_id(youtube_url)
        if not vid:
            st.error("Please paste a valid YouTube link or the 11-character video ID, or paste the transcript text.")
            st.stop()

        if debug_transcripts:
            with st.expander("DEBUG: Available transcripts for this video"):
                ts = list_available_transcripts(vid)
                if ts is None:
                    st.write("No transcript list available (captions disabled, restricted, or inaccessible).")
                else:
                    rows = [{"language_code": t.language_code, "is_generated": t.is_generated} for t in ts.transcripts]
                    st.dataframe(pd.DataFrame(rows))

        with st.spinner("Fetching transcript…"):
            raw_text = fetch_transcript_text(vid, tuple(preferred_langs), allow_any_fallback=allow_any_fallback)

        if not raw_text:
            st.warning(
                "No transcript available (captions disabled/missing/restricted). "
                "Paste the transcript text manually, or choose a different video."
            )
            st.stop()

    transcript = normalize(raw_text)

    # 2) LLM → flashcards
    with st.spinner("Asking the LLM to curate vocabulary and build flashcards…"):
        try:
            cards = call_llm_for_flashcards(transcript, max_cards, level, temperature=temperature)
        except Exception as e:
            st.error(str(e))
            st.stop()

    # 3) Display
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

    # 4) Download CSV (Anki Front/Back)
    st.download_button(
        label="⬇️ Download CSV for Anki (Front=term_it, Back=definition+examples)",
        data=df_to_csv_bytes(df),
        file_name="italian_flashcards.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "If a video lacks accessible captions, paste the transcript text manually. "
    "Use the sidebar debug option to see which transcript languages are detected."
)
