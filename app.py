"""Streamlit translation app.

Required packages:
  streamlit
  requests
  openai
  anthropic

Optional (for local token counting fallback):
  tiktoken

Secrets expected (Streamlit secrets or environment variables):
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  APP_PASSWORD   (login now uses this single password; APP_USERNAME is no longer read)
"""

from __future__ import annotations

import csv
import io
import os
import re
import sqlite3
import time
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import requests
import streamlit as st


def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("Login")

    # Single password field, wrapped in a form so that pressing Enter (or the
    # "Go"/"Done" key on a mobile keyboard) submits. A plain st.button does not
    # register as clicked when the user hits Enter, which looked like "nothing
    # happens" on some devices.
    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        # Coerce the secret to str (a purely numeric value is parsed as an int)
        # and strip surrounding whitespace, which password managers, autofill,
        # and copy-paste frequently append.
        expected = str(st.secrets["APP_PASSWORD"]).strip()
        if password.strip() == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid password")

    return False


if not check_password():
    st.stop()


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "token_usage.sqlite3"

PROMPTS_DIR = APP_DIR / "prompts"

ADVANCED_PROMPT_SOURCE = (
    "https://docs.google.com/document/d/1A-ycuuiAaKIoEsBQ6a2mb_c7Piob45fbZ1mr4spz7oo/edit?tab=t.0"
)

LANGUAGES = ["Russian"]

# --- Glossary -------------------------------------------------------------
# A glossary is a Google Sheet with three columns:
#   A = English term
#   B = its translation
#   C = optional notes for that term (context, grammatical gender, when it
#       applies) - free-form guidance passed through to the model verbatim.
#
# A sheet can hold hundreds of entries, so the whole thing is never sent. Only
# rows whose English term actually occurs in the text being translated are
# appended to the prompt, which keeps it small and stops unrelated entries
# diluting the model's attention. See select_glossary_entries().

GLOSSARY_NONE = "None"
GLOSSARY_TEAM = "Team glossary (recommended)"
GLOSSARY_CUSTOM = "Custom glossary"

# The shared team glossary. Read live (cached 60s) rather than baked in, so
# edits to the sheet take effect without a redeploy.
TEAM_GLOSSARY_SOURCE = (
    "https://docs.google.com/spreadsheets/d/1DE994NrkCdqQ9MODykrFhJ5QlK_jwv_BIkEZDm4P9vE/edit"
)

GLOSSARY_FORMAT_HELP = (
    "A Google Sheet with three columns - A: the English term, B: its translation, "
    "C: optional notes for that term (context, gender, when it applies). "
    "A header row is fine. Share it as 'Anyone with the link -> Viewer'. "
    "Only terms that actually appear in your text get added to the prompt."
)

# First-row values that mean the sheet has a header rather than a real entry.
_GLOSSARY_HEADER_TERMS = {
    "english", "english word", "english term", "term", "source", "word", "en", "isha",
}
_GLOSSARY_HEADER_META = {
    "translation", "translations", "russian", "target", "meaning",
    "gender", "note", "notes", "comment", "comments", "clarification", "context",
}

# Reasoning-effort labels shown in the UI, ordered fastest/cheapest -> most thorough.
# EFFORT_OFF turns thinking/reasoning off entirely; the rest map to API values below.
EFFORT_OFF = "Off (no thinking)"
EFFORT_API_VALUE = {
    "Low": "low",
    "Medium": "medium",
    "High": "high",
}

# Options stop at "High" (the API's own default); the deeper xhigh/max levels are
# intentionally not offered. The app itself defaults to "Off", so translations stay
# fast unless more reasoning is asked for. "Off" sits in this same list rather than
# being a separate toggle, keeping thinking a single control.
FULL_EFFORTS = [EFFORT_OFF, "Low", "Medium", "High"]

MODEL_CHOICES = {
    "Claude Opus 5": {
        "provider": "anthropic",
        "model": "claude-opus-5",
        "efforts": FULL_EFFORTS,
        "default_effort": EFFORT_OFF,
        "speed": 15.0,
    },
    "Claude Opus 4.8": {
        "provider": "anthropic",
        "model": "claude-opus-4-8",
        "efforts": FULL_EFFORTS,
        "default_effort": EFFORT_OFF,
        "speed": 15.0,
    },
    "Claude Sonnet 5": {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "efforts": FULL_EFFORTS,
        "default_effort": EFFORT_OFF,
        "speed": 45.0,
    },
    "GPT 5.6 Sol": {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "efforts": FULL_EFFORTS,
        "default_effort": EFFORT_OFF,
        "speed": 30.0,
    },
    "GPT 5.6 Terra": {
        "provider": "openai",
        "model": "gpt-5.6-terra",
        "efforts": FULL_EFFORTS,
        "default_effort": EFFORT_OFF,
        "speed": 45.0,
    },
    # GPT 5.5 predates the 5.6 effort ladder; its supported levels aren't verified
    # here, so it runs at the model default with no effort parameter sent.
    "GPT 5.5": {
        "provider": "openai",
        "model": "gpt-5.5",
        "efforts": [],
        "default_effort": None,
        "speed": 50.0,
    },
}

# Rough throughput multipliers for the ETA only — higher effort means more thinking
# tokens and a slower wall clock. These are heuristics, like the base speeds.
EFFORT_SPEED_FACTOR = {
    EFFORT_OFF: 1.6,
    "Low": 1.3,
    "Medium": 1.0,
    "High": 0.75,
}


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL
            )
            """
        )


def save_operation(provider: str, model: str, input_tokens: int, output_tokens: int) -> int:
    total_tokens = int(input_tokens) + int(output_tokens)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO operations (created_at, provider, model, input_tokens, output_tokens, total_tokens)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                provider,
                model,
                int(input_tokens),
                int(output_tokens),
                total_tokens,
            ),
        )
    return total_tokens


def get_total_tokens_all_time() -> int:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT COALESCE(SUM(total_tokens), 0) FROM operations").fetchone()
    return int(row[0] or 0)


def get_token_count_fallback(text: str) -> int:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text.split()))


def extract_google_doc_id(url: str) -> Optional[str]:
    if not url:
        return None
    match = re.search(r"/document/d/([a-zA-Z0-9-_]+)", url)
    return match.group(1) if match else None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_public_google_doc_text(url: str) -> str:
    doc_id = extract_google_doc_id(url)
    if not doc_id:
        raise ValueError("Could not find a Google Doc ID in the link.")

    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    response = requests.get(export_url, timeout=30)
    response.raise_for_status()
    return response.text.strip()


def load_bundled_prompt(filename: str) -> str:
    """Read a prompt shipped alongside the app.

    Nastya's prompt lives in the repo rather than being fetched from its Google
    Doc: that doc is not publicly viewable, and the app authenticates to Google
    nowhere, so an export request returns 401. Reading it locally is also faster
    than a network round trip. To update it, edit the file and redeploy.
    """
    path = PROMPTS_DIR / filename
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RuntimeError(f"Bundled prompt file is missing: {path}")


def extract_google_sheet_id(url: str) -> Optional[str]:
    if not url:
        return None
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return match.group(1) if match else None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_google_sheet_rows(url: str) -> list:
    """Read a public Google Sheet as a list of (term, translation, notes).

    Cached for only a minute: a glossary is a living document, so edits should
    show up quickly without a redeploy.
    """
    sheet_id = extract_google_sheet_id(url)
    if not sheet_id:
        raise ValueError("Could not find a Google Sheet ID in that link.")

    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    gid = re.search(r"[#&?]gid=([0-9]+)", url)
    if gid:
        export_url += f"&gid={gid.group(1)}"

    response = requests.get(export_url, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"

    rows = []
    for raw in csv.reader(io.StringIO(response.text)):
        term = (raw[0] if len(raw) > 0 else "").strip()
        translation = (raw[1] if len(raw) > 1 else "").strip()
        notes = (raw[2] if len(raw) > 2 else "").strip()
        if term and translation:
            rows.append((term, translation, notes))

    # Drop a header row. Checking columns B/C as well as A matters: a sheet may
    # label column A with something domain-specific ("isha") that no generic
    # list would catch, while B/C are almost always "Translation"/"Gender".
    if rows:
        term, translation, notes = rows[0]
        if (
            term.lower() in _GLOSSARY_HEADER_TERMS
            or translation.lower() in _GLOSSARY_HEADER_META
            or notes.lower() in _GLOSSARY_HEADER_META
        ):
            rows = rows[1:]
    return rows


# Endings an English term may pick up in running text.
_INFLECTION = r"(?:e|es|s|ed|d|ing|ion|ions|al|ally|ly|'s|’s)?"


def build_term_pattern(term: str) -> Optional[str]:
    """Regex matching a glossary term regardless of case or inflected ending.

    Matching is deliberately generous: a spurious entry costs one line of prompt,
    while a missed one silently loses the required translation.
    """
    words = re.findall(r"[A-Za-z0-9']+", term)
    if not words:
        return None

    parts = []
    for word in words:
        base = word.lower()
        # Drop a final 'e' so "meditate" also matches "meditating"/"meditation";
        # the suffix group restores it for the base form.
        if len(base) > 3 and base.endswith("e"):
            base = base[:-1]
        stem = re.escape(base)
        # Allow the final consonant to double, so "run" matches "running".
        if (
            len(base) >= 3
            and base[-1].isalpha()
            and base[-1] not in "aeiou"
            and base[-2] in "aeiou"
        ):
            stem += re.escape(base[-1]) + "?"
        parts.append(stem + _INFLECTION)

    # Join with non-word characters rather than plain whitespace: multi-word
    # entries in a real glossary contain commas, hyphens and line breaks.
    return r"\b" + r"\W+".join(parts) + r"\b"


def select_glossary_entries(rows: list, text: str) -> list:
    """Keep only the glossary rows whose English term appears in `text`."""
    if not rows or not text:
        return []
    matched = []
    for row in rows:
        pattern = build_term_pattern(row[0])
        if not pattern:
            continue
        try:
            if re.search(pattern, text, re.IGNORECASE):
                matched.append(row)
        except re.error:
            continue
    return matched


def format_glossary_block(entries: list) -> str:
    lines = [
        "",
        "---",
        "",
        "GLOSSARY - required terminology for this text",
        "",
        "Use these translations for the English terms listed below. Where a note is",
        "given it is context-specific guidance for that term (for example which",
        "grammatical gender to use); apply it in preference to the default form.",
        "",
    ]
    for term, translation, notes in entries:
        line = '- "' + term + '" -> "' + translation + '"'
        if notes:
            line += "  [" + notes + "]"
        lines.append(line)
    return "\n".join(lines)


def build_glossary_section(glossary_mode: str, custom_url: str, text: str):
    """Return (block, matched_count, total_count) for the selected glossary."""
    if glossary_mode == GLOSSARY_NONE:
        return "", 0, 0

    if glossary_mode == GLOSSARY_TEAM:
        url = TEAM_GLOSSARY_SOURCE
        if not url:
            raise RuntimeError("The team glossary link is not configured.")
    else:
        url = custom_url
        if not url.strip():
            raise RuntimeError("Paste a Google Sheet link for the custom glossary.")

    rows = fetch_google_sheet_rows(url)
    matched = select_glossary_entries(rows, text)
    if not matched:
        return "", 0, len(rows)
    return format_glossary_block(matched), len(matched), len(rows)


def resolve_prompt(prompt_mode: str, language: str, prompt_doc_url: str, custom_prompt: str) -> str:
    if prompt_mode == "Simple prompt":
        return f"Translate the following into {language}"
    if prompt_mode == "Advanced (Yury)":
        # Still read live from its Google Doc, which is publicly viewable, so
        # edits to that doc take effect without a redeploy (cached 5 min).
        prompt_text = fetch_public_google_doc_text(ADVANCED_PROMPT_SOURCE)
        return prompt_text.replace("$LANGUAGE", language)
    if prompt_mode == "Advanced (Nastya)":
        prompt_text = load_bundled_prompt("advanced_nastya.md")
        return prompt_text.replace("$LANGUAGE", language)
    if prompt_mode == "Custom (Google Doc)":
        prompt_text = fetch_public_google_doc_text(prompt_doc_url)
        return prompt_text.replace("$LANGUAGE", language)
    if prompt_mode == "Custom (Paste)":
        return custom_prompt.strip().replace("$LANGUAGE", language)
    raise ValueError("Unknown prompt mode.")


def resolve_input(input_mode: str, pasted_text: str, input_doc_url: str) -> str:
    if input_mode == "Paste text":
        return pasted_text.strip()
    if input_mode == "Google Doc":
        return fetch_public_google_doc_text(input_doc_url)
    raise ValueError("Unknown input mode.")


def chunk_text(text: str, max_words: int = 10000) -> list[str]:
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_word_count = 0

    for p in paragraphs:
        p_word_count = len(p.split())
        if current_word_count + p_word_count > max_words and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [p]
            current_word_count = p_word_count
        else:
            current_chunk.append(p)
            current_word_count += p_word_count

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
        
    return chunks


def call_model(
    provider: str,
    model: str,
    prompt: str,
    user_text: str,
    api_key: str,
    effort: Optional[str] = None,
) -> Tuple[str, int, int]:
    # max_tokens covers thinking *and* the translation. 30k is ample now that
    # effort stops at "High"; the deeper levels that needed 64k aren't offered.
    MAX_OUTPUT_LIMIT = 30000

    api_effort = EFFORT_API_VALUE.get(effort) if effort else None

    if provider == "anthropic":
        from anthropic import Anthropic  # type: ignore
        client = Anthropic(api_key=api_key)

        # Thinking blocks are filtered out of output_text below, so reasoning never
        # leaks into the translation itself.
        request_kwargs = {}
        if effort == EFFORT_OFF:
            request_kwargs["thinking"] = {"type": "disabled"}
        elif api_effort:
            request_kwargs["thinking"] = {"type": "adaptive"}
            request_kwargs["output_config"] = {"effort": api_effort}

        with client.messages.stream(
            model=model,
            max_tokens=MAX_OUTPUT_LIMIT,
            system=prompt,
            messages=[{"role": "user", "content": user_text}],
            **request_kwargs,
        ) as stream:
            response = stream.get_final_message()

        output_text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        
        if not input_tokens: input_tokens = get_token_count_fallback(prompt + "\n" + user_text)
        if not output_tokens: output_tokens = get_token_count_fallback(output_text)
        
        return output_text.strip(), input_tokens, output_tokens

    if provider == "openai":
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        
        # GPT 5.6 uses "none" as its no-reasoning setting. Models with no configured
        # effort list (e.g. GPT 5.5) send no reasoning parameter at all.
        request_kwargs = {}
        if effort == EFFORT_OFF:
            request_kwargs["reasoning"] = {"effort": "none"}
        elif api_effort:
            request_kwargs["reasoning"] = {"effort": api_effort}

        response = client.responses.create(
            model=model,
            instructions=prompt,
            input=user_text,
            max_output_tokens=MAX_OUTPUT_LIMIT,
            **request_kwargs,
        )
        
        output_text = getattr(response, "output_text", None)
        if not output_text:
            output_parts = []
            for item in getattr(response, "output", []) or []:
                for content in getattr(item, "content", []) or []:
                    text = getattr(content, "text", None)
                    if text: output_parts.append(text)
            output_text = "".join(output_parts)

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        
        if not input_tokens: input_tokens = get_token_count_fallback(prompt + "\n" + user_text)
        if not output_tokens: output_tokens = get_token_count_fallback(output_text)
        
        return output_text.strip(), input_tokens, output_tokens

    raise ValueError(f"Unsupported provider: {provider}")


class TranslationThread(threading.Thread):
    def __init__(self, provider, model, prompt, user_text, api_key, effort=None):
        super().__init__()
        self.provider = provider
        self.model = model
        self.prompt = prompt
        self.user_text = user_text
        self.api_key = api_key
        self.effort = effort
        self.result = None
        self.exception = None

    def run(self):
        try:
            self.result = call_model(
                self.provider, self.model, self.prompt, self.user_text,
                self.api_key, self.effort,
            )
        except Exception as e:
            self.exception = e


# --- Main UI ---
st.set_page_config(page_title="Streamlit Translator", layout="wide")
init_db()

st.title("Streamlit Translator")
st.caption("Translates pasted text or a public Google Doc into Russian.")

top_left, top_mid, top_right = st.columns([1, 1, 1])
with top_left:
    language = st.selectbox("Select language", LANGUAGES, index=0)
with top_mid:
    model_label = st.selectbox("Select model", list(MODEL_CHOICES.keys()), index=0)

model_info = MODEL_CHOICES[model_label]

with top_right:
    # The option list is per-model. Keying the widget by model label gives each
    # model its own selection, so switching models can't carry over a level the
    # new model doesn't support.
    available_efforts = model_info["efforts"]
    if available_efforts:
        effort = st.selectbox(
            "Reasoning effort",
            available_efforts,
            index=available_efforts.index(model_info["default_effort"]),
            key=f"effort_{model_label}",
            help="Higher levels think longer before translating: better on nuanced "
                 "passages, but slower and more tokens. 'Off' skips thinking entirely.",
        )
    else:
        st.selectbox(
            "Reasoning effort",
            ["Model default"],
            disabled=True,
            key=f"effort_{model_label}",
            help="This model has no configurable reasoning effort.",
        )
        effort = None

st.divider()

prompt_mode = st.radio(
    "Prompt selection",
    [
        "Simple prompt",
        "Advanced (Yury)",
        "Advanced (Nastya)",
        "Custom (Google Doc)",
        "Custom (Paste)",
    ],
    horizontal=True,
)

prompt_doc_url = ""
custom_prompt = ""

if prompt_mode == "Custom (Google Doc)":
    prompt_doc_url = st.text_input(
        "Google Doc link for the prompt",
        placeholder="https://docs.google.com/document/d/.../edit",
        help="Make sure the Google Doc is set to View for everyone.",
    )
elif prompt_mode == "Custom (Paste)":
    custom_prompt = st.text_area(
        "Custom prompt",
        height=180,
        placeholder="Type or paste your prompt here.",
    )

# Read-only preview of the prompt that will actually be sent, so the effect of the
# selection above is visible without running a translation. Skipped for
# Custom (Paste), where the editable box above already shows the same text.
if prompt_mode != "Custom (Paste)":
    if prompt_mode == "Custom (Google Doc)" and not prompt_doc_url.strip():
        prompt_preview = "Paste a Google Doc link above to preview the prompt."
    else:
        try:
            prompt_preview = resolve_prompt(prompt_mode, language, prompt_doc_url, custom_prompt)
        except Exception as exc:
            # An unreachable Doc must not break the page.
            prompt_preview = f"Preview unavailable: {exc}"

    st.text_area(
        "Prompt preview",
        value=prompt_preview,
        height=120,
        disabled=True,
        help="Read-only. This is the prompt that will be sent along with your text.",
    )

glossary_mode = st.radio(
    "Glossary",
    [GLOSSARY_NONE, GLOSSARY_TEAM, GLOSSARY_CUSTOM],
    index=1 if TEAM_GLOSSARY_SOURCE else 0,
    horizontal=True,
    help=GLOSSARY_FORMAT_HELP,
)

custom_glossary_url = ""
if glossary_mode == GLOSSARY_CUSTOM:
    custom_glossary_url = st.text_input(
        "Google Sheet link for the glossary",
        placeholder="https://docs.google.com/spreadsheets/d/.../edit",
        help=GLOSSARY_FORMAT_HELP,
    )
    st.caption(
        "Three columns: **A** the English term, **B** its translation, "
        "**C** optional notes for that term - context, grammatical gender, or when "
        "it applies. A header row is fine. The sheet must be shared as "
        "'Anyone with the link -> Viewer'."
    )

if glossary_mode != GLOSSARY_NONE:
    # Points at the sheet in use rather than showing its contents: the sheet runs
    # to hundreds of entries, and only the handful matching the text ever reach
    # the model. Those are visible in the full prompt shown after a run.
    if glossary_mode == GLOSSARY_TEAM:
        glossary_preview = TEAM_GLOSSARY_SOURCE or "The team glossary link is not configured."
    else:
        glossary_preview = custom_glossary_url.strip() or "Paste a Google Sheet link above."

    st.text_area(
        "Glossary preview",
        value=glossary_preview,
        height=68,
        disabled=True,
        help=(
            "The glossary sheet in use. Its contents are not shown here - only the "
            "entries whose English term appears in your text are added to the "
            "prompt, and you can see exactly which ones after a translation runs."
        ),
    )
    st.caption(
        "Only entries whose English term actually appears in your text are added "
        "to the prompt, so a large glossary stays cheap. Matching ignores case and "
        "common word endings."
    )

st.subheader("Input text")
input_mode = st.radio("Input source", ["Paste text", "Google Doc"], horizontal=True)

pasted_text = ""
input_doc_url = ""
if input_mode == "Paste text":
    pasted_text = st.text_area(
        "Paste text",
        height=240,
        placeholder="Paste the text to translate here.",
    )
    if len(pasted_text.split()) > 3000:
        st.warning("You are about to translate a very large document.")
else:
    input_doc_url = st.text_input(
        "Google Doc link for input text",
        placeholder="https://docs.google.com/document/d/.../edit",
        help="Make sure the Google Doc is set to View for everyone.",
    )

if "is_translating" not in st.session_state:
    st.session_state.is_translating = False

def start_translation():
    st.session_state.is_translating = True
    # Drop any error from a previous attempt so the banner reflects this run.
    st.session_state.pop("last_error", None)
    st.session_state.pop("last_error_detail", None)

st.button(
    "Translate", 
    type="primary", 
    use_container_width=False, 
    disabled=st.session_state.is_translating,
    on_click=start_translation
)

if st.session_state.is_translating:
    try:
        # Input is resolved first: which glossary entries are relevant depends on
        # the text, so the prompt can only be finished once the text is known.
        input_text = resolve_input(input_mode, pasted_text, input_doc_url)
        prompt_text = resolve_prompt(prompt_mode, language, prompt_doc_url, custom_prompt)

        if not input_text:
            st.error("Input text is empty.")
        else:
            if len(input_text.split()) > 3000 and input_mode == "Google Doc":
                st.warning("You are about to translate a very large document.")

            glossary_block, glossary_hits, glossary_total = build_glossary_section(
                glossary_mode, custom_glossary_url, input_text
            )
            if glossary_block:
                prompt_text = prompt_text.rstrip() + "\n" + glossary_block
            if glossary_total:
                st.caption(
                    f"Glossary: {glossary_hits} of {glossary_total} entries matched "
                    f"your text and were added to the prompt."
                )

            chunks = chunk_text(input_text, max_words=10000)
            
            full_output_text = ""
            total_input_tokens = 0
            total_output_tokens = 0

            anthropic_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

            # Base throughput comes from the model registry; deeper reasoning
            # slows the wall clock, so scale by the selected effort.
            avg_speed = model_info["speed"] * EFFORT_SPEED_FACTOR.get(effort, 1.0)

            est_words = len(input_text.split())
            est_total_tokens = est_words * 2.8 
            est_total_seconds = max(5.0, est_total_tokens / avg_speed)

            progress_bar = st.progress(0, text=f"Translating document (0/{len(chunks)} chunks)... Estimated time: ~{int(est_total_seconds)}s")
            start_time = time.time()

            for i, chunk in enumerate(chunks):
                active_api_key = anthropic_key if model_info["provider"] == "anthropic" else openai_key
                if not active_api_key:
                    raise RuntimeError(f"Missing API key for {model_info['provider']}.")

                t = TranslationThread(model_info["provider"], model_info["model"], prompt_text, chunk, active_api_key, effort)
                t.start()
                
                chunk_start_time = time.time()
                chunk_est_seconds = est_total_seconds / len(chunks)

                while t.is_alive():
                    elapsed = time.time() - chunk_start_time
                    chunk_progress = min(elapsed / chunk_est_seconds, 0.95)
                    
                    overall_progress = (i + chunk_progress) / len(chunks)
                    overall_progress = min(overall_progress, 1.0)
                    
                    remaining_overall = max(0, int(est_total_seconds - (time.time() - start_time)))
                    
                    progress_bar.progress(overall_progress, text=f"Translating chunk {i+1} of {len(chunks)}... (ETA: ~{remaining_overall}s)")
                    time.sleep(0.2)

                t.join()
                
                if t.exception:
                    raise t.exception

                chunk_output, in_tokens, out_tokens = t.result
                
                full_output_text += chunk_output + "\n\n"
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens

            progress_bar.progress(1.0, text="Translation complete!")

            total_tokens = save_operation(
                model_info["provider"],
                model_info["model"],
                total_input_tokens,
                total_output_tokens,
            )
            all_time_tokens = get_total_tokens_all_time()

            st.session_state["last_output"] = full_output_text.strip()
            st.session_state["last_prompt"] = prompt_text
            st.session_state["last_input"] = input_text
            st.session_state["last_input_tokens"] = total_input_tokens
            st.session_state["last_output_tokens"] = total_output_tokens
            st.session_state["last_total_tokens"] = total_tokens
            st.session_state["last_all_time_tokens"] = all_time_tokens
            st.session_state["last_model_label"] = model_label
            st.session_state["last_effort"] = effort or "Model default"
            st.session_state["last_glossary"] = (
                f"{glossary_mode} ({glossary_hits}/{glossary_total} entries used)"
                if glossary_total
                else glossary_mode
            )
            st.session_state["last_model_name"] = model_info["model"]
            st.session_state["last_provider"] = model_info["provider"]
            st.session_state["last_chunk_count"] = len(chunks)
            st.session_state.pop("last_error", None)
            st.session_state.pop("last_error_detail", None)
            
    except Exception as exc:
        # st.rerun() below aborts this run immediately and restarts the script,
        # discarding anything rendered here - an st.error() call included. The
        # failure has to be stashed in session state to survive that, otherwise
        # the translation fails completely silently.
        st.session_state["last_error"] = f"{type(exc).__name__}: {exc}"
        st.session_state["last_error_detail"] = traceback.format_exc()
    finally:
        st.session_state.is_translating = False
        st.rerun()

if st.session_state.get("last_error"):
    st.error(f"Translation failed. {st.session_state['last_error']}")
    with st.expander("Technical details"):
        st.code(st.session_state.get("last_error_detail", ""), language="text")

if "last_output" in st.session_state:
    st.divider()
    st.subheader("Output")
    
    st.text_area(
        label="Translated Document",
        value=st.session_state["last_output"],
        height=500,
        disabled=False,
        help="Click inside and press Ctrl+A (Cmd+A on Mac), then Ctrl+C to copy all text."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        st.download_button(
            "Download as txt",
            data=st.session_state["last_output"].encode("utf-8"),
            file_name="translation.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # Expanded stats block containing all requested info
    st.caption(
        f"**Model used:** {st.session_state['last_model_label']} ({st.session_state['last_model_name']}) | "
        f"**Reasoning effort:** {st.session_state.get('last_effort', 'n/a')} | "
        f"**Glossary:** {st.session_state.get('last_glossary', 'n/a')} | "
        f"**Chunks processed:** {st.session_state.get('last_chunk_count', 1)}"
    )
    st.caption(
        f"This run used about {st.session_state['last_input_tokens'] + st.session_state['last_output_tokens']} tokens "
        f"({st.session_state['last_input_tokens']} input + {st.session_state['last_output_tokens']} output)."
    )
    st.caption(f"All-time token total stored locally: {st.session_state['last_all_time_tokens']}.")

    # The exact prompt that went to the model, glossary block included, so the
    # selected terminology is inspectable after the fact.
    sent_prompt = st.session_state.get("last_prompt", "")
    st.text_area(
        "Prompt sent to the model",
        value=sent_prompt,
        height=300,
        disabled=True,
        help="Read-only. The complete system prompt for this run, including any "
             "glossary entries matched against your text.",
    )
    st.caption(
        f"Full prompt: {len(sent_prompt.split())} words, {len(sent_prompt)} characters. "
        f"Glossary: {st.session_state.get('last_glossary', 'n/a')}."
    )

else:
    st.caption("No translation has been run yet.")

st.divider()
st.caption("")
