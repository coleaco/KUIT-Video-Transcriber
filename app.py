
import os
import re
import json
from urllib.parse import urlparse, parse_qs
from typing import List, Dict

import streamlit as st
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# ========= App Config =========
APP_TITLE = "Italian Vocab from YouTube — LLM-Only"
MAX_CONTEXT_CHARS = 12000   # cap transcript length sent to LLM
DEFAULT_MAX_CARDS = 18

# ========= Helpers =========

def extract_video_id(url_or_id: str) -> str | None:
    """Extract an 11-char YouTube ID from a URL, or accept a bare ID."""
    url_or_id = url_or_id.strip()
    if re.match(r"^[\w-]{11}$", url_or_id):
        return url_or_id
    try:
        parsed = urlparse(url_or_id)
        if parsed.netloc in ("youtu.be", "www.youtu.be"):
            vid = parsed.path.lstrip("/")
            return vid if re.match(r"^[\w-]{11}$", vid) else None
        if parsed.netloc in ("youtube.com", "www.youtube.com", "m.youtube.com"):
            if parsed.path == "/watch":
                v = parse_qs(parsed.query).get("v", [None])[0]
                return v if v and re.match(r"^[\w-]{11}$", v) else None
            if parsed.path.startswith("/shorts/"):
                parts = [p for p in parsed.path.split("/") if p]
                vid = parts[1] if len(parts) > 1 else None
                return vid if vid and re.match(r"^[\w-]{11}$", vid) else None
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def fetch_transcript_text(video_id: str, preferred_langs=('it','it-IT','en')) -> str | None:
    """
    Try to fetch a transcript (manual or auto captions) in preferred language order.
    No API key is required for youtube-transcript-api.
    """
    try:
        for lang in preferred_langs:
            try:
                entries = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                return " ".join([e["text"] for e in entries])
            except NoTranscriptFound:
                continue
        return None
    except TranscriptsDisabled:
        return None
    except Exception:
        return None

def normalize(text: str) -> str:
    text = text.replace("\u200b", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def build_flashcard_prompt(transcript: str, max_items: int, level: str) -> str:
    """
    Ask the LLM to select the most pedagogically useful vocabulary
    and return STRICT JSON for easy parsing and CSV export.
    """
    # Trim to avoid excessive token usage; we still give rich context
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
    "term_it": "…",            // the Italian word/expression exactly as it appears
    "pos": "…",                // part of speech (e.g., verb, noun, adj, expr)
    "translation_en": "…",     // concise English gloss/translation
    "example_it": "…",         // simple example sentence in Italian (A2–B1)
    "gloss_en": "…",           // English gloss of the example
    "rationale": "…"           // 1-line note on pedagogical/value/difficulty
  }},
  ...
]

Transcript:
{transcript}
"""

def call_llm_for_flashcards(transcript: str, max_items: int, level: str, temperature: float = 0.3) -> List[Dict]:
    """
    Uses an OpenAI-compatible Chat Completions API (via OPENAI_API_KEY in secrets).
    Returns a list of flashcard dicts, or raises an exception with an informative message.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit Secrets. Add it via the app menu: ⋮ → Edit secrets.")

    # Lazy import to keep the app working even if the package isn't installed locally
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    prompt = build_flashcard_prompt(transcript, max_items, level)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
    )
    raw = resp.choices[0].message.content.strip()

    # Try to parse JSON. If it fails, raise an informative error showing a snippet.
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("LLM response is not a JSON list.")
        # Basic schema check
        required = {"term_it","pos","translation_en","example_it","gloss_en","rationale"}
        for i, item in enumerate(data):
            if not isinstance(item, dict) or not required.issubset(item.keys()):
                raise ValueError(f"Item {i} missing required keys. Found keys: {list(item.keys())}")
        return data
    except Exception as e:
        # Allow the user to see raw output if JSON parse fails
        raise ValueError(f"Could not parse LLM JSON. Error: {e}\nRaw output (first 800 chars): {raw[:800]}")

def flashcards_to_dataframe(cards: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(cards)
    # Ensure consistent column order
    cols = ["term_it","pos","translation_en","example_it","gloss_en","rationale"]
    return df[cols]

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    # Export Front/Back for Anki and keep a richer CSV as well
    export_basic = pd.DataFrame({
        "Front": df["term_it"],
        "Back": df.apply(lambda r: f"({r['pos']}) — {r['translation_en']}<br><br>"
                                   f"IT: {r['example_it']}<br>"
                                   f"EN: {r['gloss_en']}", axis=1)
    })
    return export_basic.to_csv(index=False).encode("utf-8")


# ========= UI =========

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.markdown(
    "Paste a **YouTube** link (or a transcript). The app fetches captions when available, "
    "sends the text to an LLM, and returns **curated vocabulary flashcards**—no frequency list needed."
)

# Sidebar controls
st.sidebar.header("Settings")
max_cards = st.sidebar.slider("Number of flashcards", 6, 30, DEFAULT_MAX_CARDS, step=2)
level = st.sidebar.selectbox("Learner level focus", ["A2", "B1", "B2"])
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)
show_rationales = st.sidebar.checkbox("Show rationale/why this item was chosen", value=False)

st.sidebar.markdown(
    "**Tip:** Add your `OPENAI_API_KEY` (and optional `OPENAI_MODEL`) via the app menu **⋮ → Edit secrets**."
)

# Inputs
youtube_url = st.text_input("YouTube URL (or 11-character ID)")
manual_text = st.text_area("Or paste the Italian transcript here (overrides URL if filled)", height=180)

if st.button("Generate flashcards"):
    # 1) Get transcript text
    if manual_text.strip():
        raw_text = manual_text.strip()
    else:
        vid = extract_video_id(youtube_url)
        if not vid:
            st.error("Please paste a valid YouTube link or an 11-character video ID, or paste the transcript.")
            st.stop()
        with st.spinner("Fetching transcript…"):
            raw_text = fetch_transcript_text(vid)
        if not raw_text:
            st.warning("No transcript available (captions disabled or missing). Paste the transcript text manually, or choose a different video.")
            st.stop()

    transcript = normalize(raw_text)

    # 2) LLM: select vocab & produce flashcards
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

    # 4) Download CSV for Anki (Front/Back)
    st.download_button(
        label="⬇️ Download CSV for Anki (Front=term_it, Back=definition+examples)",
        data=df_to_csv_bytes(df),
        file_name="italian_flashcards.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Transcripts are retrieved via the open-source `youtube-transcript-api` (manual or auto captions; no YouTube API key required). "
    "If a video lacks captions, paste the transcript text manually."
)
