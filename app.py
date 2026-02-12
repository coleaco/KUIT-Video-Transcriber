import os
import re
import io
from urllib.parse import urlparse, parse_qs

import streamlit as st
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from collections import Counter

# ---------------------------
# Configuration
# ---------------------------
APP_TITLE = "Italian Vocab from YouTube (500–20,000 rank)"
DEFAULT_MIN_RANK = 500
DEFAULT_MAX_RANK = 20000
MAX_CANDIDATES_DEFAULT = 20

# Minimal Italian stopword set (you can expand this or load from a file)
ITALIAN_STOPWORDS = {
    "di","a","da","in","con","su","per","tra","fra","e","o","ma","anche","che","come","dove","quando","perché","piu","più","meno",
    "il","lo","la","i","gli","le","un","uno","una","mi","ti","si","ci","vi","ne","dei","degli","delle","del","dello","della","dai",
    "dalle","dagli","dall","al","allo","alla","ai","agli","alle","questo","questa","questi","queste","quello","quella","quelli",
    "quelle","qui","lì","là","quindi","se","non","c","l","d","s"
}

# ---------------------------
# Helper functions
# ---------------------------

def extract_video_id(url_or_id: str) -> str | None:
    """Extract an 11-char YouTube video ID from a URL or return the input if it already looks like an ID."""
    url_or_id = url_or_id.strip()
    # If user pasted a bare ID
    if re.match(r"^[\w-]{11}$", url_or_id):
        return url_or_id

    try:
        parsed = urlparse(url_or_id)
        # youtu.be/<id>
        if parsed.netloc in ("youtu.be", "www.youtu.be"):
            vid = parsed.path.lstrip("/")
            return vid if re.match(r"^[\w-]{11}$", vid) else None
        # youtube.com/watch?v=<id>
        if parsed.netloc in ("youtube.com", "www.youtube.com", "m.youtube.com"):
            if parsed.path == "/watch":
                v = parse_qs(parsed.query).get("v", [None])[0]
                return v if v and re.match(r"^[\w-]{11}$", v) else None
            # /shorts/<id>
            if parsed.path.startswith("/shorts/"):
                vid = parsed.path.split("/")[2] if len(parsed.path.split("/")) > 2 else parsed.path.split("/")[1]
                return vid if re.match(r"^[\w-]{11}$", vid) else None
    except Exception:
        return None
    return None

def fetch_transcript_text(video_id: str, preferred_langs=('it','it-IT','en')) -> str | None:
    """
    Try to fetch transcript in preferred languages order.
    Works with manually provided or auto-generated captions and requires no API key.
    Returns raw joined text or None if unavailable.
    """
    try:
        # Try the simple API first
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
    text = text.lower()
    # remove timestamps (already not present) & extra whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_words(text: str) -> list[str]:
    # Keep Italian accented letters as word chars
    words = re.findall(r"[a-zA-ZàèéìòóùÀÈÉÌÒÓÙ]+", text, flags=re.UNICODE)
    return [w.lower() for w in words]

def load_frequency_list(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect columns: word, rank
    df["word"] = df["word"].astype(str).str.lower()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["rank"])
    return df

def filter_by_rank(tokens: list[str], freq_df: pd.DataFrame, min_rank: int, max_rank: int) -> pd.DataFrame:
    counts = Counter([t for t in tokens if t not in ITALIAN_STOPWORDS])
    if not counts:
        return pd.DataFrame(columns=["word","count","rank"])

    cand_df = pd.DataFrame(counts.items(), columns=["word", "count"])
    merged = cand_df.merge(freq_df, on="word", how="left")
    # Keep those with rank in the desired band
    band = merged[(merged["rank"] >= min_rank) & (merged["rank"] <= max_rank)]
    # If rank missing, drop
    band = band.dropna(subset=["rank"])
    # Strong default sort: first by descending in-text count, then ascending rank
    band = band.sort_values(by=["count", "rank"], ascending=[False, True])
    return band

def make_flashcard_prompt(word: str) -> str:
    return f"""Create a concise Italian vocabulary flashcard for: {word}
Include:
- Italian word
- Part of speech
- English translation
- One simple Italian example sentence (A2–B1)
- English gloss of that sentence
Format as:
**{word}** (POS) — <English translation>
IT: <Italian example>
EN: <English gloss>
"""

def generate_flashcard_with_llm(word: str, transcript_context: str | None = None) -> str:
    """
    Uses OpenAI-compatible Chat Completions if OPENAI_API_KEY is present in Streamlit secrets.
    If no key, returns a templated fallback card.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")  # you can change this in Streamlit secrets
    if not api_key:
        # Fallback card (no external call)
        return f"**{word}** (POS) — <translation>\nIT: <example sentence>\nEN: <gloss>\n"

    try:
        # Import here so app still runs without openai installed (when user doesn't toggle LLM)
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        base_prompt = make_flashcard_prompt(word)
        if transcript_context:
            base_prompt += f"\nContext (excerpt from transcript):\n{transcript_context[:800]}"

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": base_prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"LLM error for '{word}': {e}")
        return f"**{word}** (POS) — <translation>\nIT: <example sentence>\nEN: <gloss>\n"

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Export two columns CSV suitable for flashcard tools (Front/Back).
    Here we place the Italian word on the Front and a combined Back.
    """
    export = pd.DataFrame({
        "Front": df["word"],
        "Back": df["flashcard"]
    })
    return export.to_csv(index=False).encode("utf-8")

# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.markdown(
    "Paste a **YouTube** link, set your **frequency band**, and generate flashcards. "
    "If the video has captions (manual or auto), the transcript can be fetched without any API key. "
    "You can optionally use an LLM API key (OpenAI-compatible) to generate definitions/examples.",
)

# Sidebar settings
st.sidebar.header("Settings")
min_rank = st.sidebar.number_input("Min rank", value=DEFAULT_MIN_RANK, min_value=1, max_value=500000, step=50)
max_rank = st.sidebar.number_input("Max rank", value=DEFAULT_MAX_RANK, min_value=1, max_value=500000, step=50)
max_cards = st.sidebar.slider("Max flashcards to generate", 5, 100, MAX_CANDIDATES_DEFAULT, step=5)
sort_choice = st.sidebar.selectbox("Sort by", ["Frequency in transcript (desc), then rank", "Alphabetical (A→Z)", "Rank (asc)"])
use_llm = st.sidebar.checkbox("Use LLM for definitions/examples (requires OPENAI_API_KEY in Secrets)", value=False)

st.sidebar.markdown("**Frequency list file expected:** `italian_frequency_list.csv` in app folder (columns: `word,rank`).")

# Frequency list loader
@st.cache_data(show_spinner=False)
def _load_freq():
    return load_frequency_list("italian_frequency_list.csv")

try:
    freq_df = _load_freq()
except FileNotFoundError:
    st.error("Could not find `italian_frequency_list.csv`. Please upload it to the app folder.")
    st.stop()

# Main controls
youtube_url = st.text_input("YouTube URL (or 11-character video ID)")
manual_text = st.text_area("Or paste Italian transcript text here (will override URL if filled)", height=180)

if st.button("Generate vocabulary"):
    raw_text = None
    context_snippet = None

    if manual_text.strip():
        raw_text = manual_text.strip()
    else:
        vid = extract_video_id(youtube_url)
        if not vid:
            st.error("Please paste a valid YouTube link or the 11-character video ID.")
            st.stop()

        with st.spinner("Fetching transcript..."):
            raw_text = fetch_transcript_text(vid)
        if not raw_text:
            st.warning("No transcript available (captions disabled or not provided). "
                       "Paste transcript text manually, or choose a different video.")
            st.stop()

    norm = normalize(raw_text)
    tokens = tokenize_words(norm)

    with st.spinner("Finding candidate vocabulary..."):
        band_df = filter_by_rank(tokens, freq_df, min_rank=min_rank, max_rank=max_rank)
        if band_df.empty:
            st.info("No words matched the specified frequency band. Try widening the range.")
            st.stop()

        # Apply sort choice
        if sort_choice == "Alphabetical (A→Z)":
            band_df = band_df.sort_values(by="word", ascending=True)
        elif sort_choice == "Rank (asc)":
            band_df = band_df.sort_values(by="rank", ascending=True)
        # else keep default

        # Limit the number of candidates
        band_df = band_df.head(max_cards)

    st.subheader("Candidate words")
    st.dataframe(band_df[["word", "count", "rank"]].reset_index(drop=True), use_container_width=True)

    # Optional context snippet for better examples
    context_snippet = norm[:1200]

    st.subheader("Flashcards")
    cards = []
    for w in band_df["word"]:
        # For performance, keep the context short
        if use_llm:
            card = generate_flashcard_with_llm(w, transcript_context=context_snippet)
        else:
            card = f"**{w}** (POS) — <translation>\nIT: <example sentence>\nEN: <gloss>\n"
        cards.append(card)
        st.markdown(card)

    export_df = band_df[["word"]].copy()
    export_df["flashcard"] = cards

    # Download CSV (Anki-friendly)
    st.download_button(
        label="⬇️ Download CSV (Front=Italian word, Back=flashcard)",
        data=df_to_csv_bytes(export_df),
        file_name="italian_flashcards.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Transcripts are obtained via the open-source **youtube-transcript-api** "
    "(works for manual or auto-generated captions; no API key required). "
    "If a video lacks captions, paste the transcript manually. "
)
