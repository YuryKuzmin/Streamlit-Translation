"""Streamlit translation stub app.

Required packages:
  streamlit
  requests
  openai
  anthropic
  google-api-python-client
  google-auth

Optional (for local token counting fallback):
  tiktoken

Secrets expected (Streamlit secrets or environment variables):
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  GOOGLE_SERVICE_ACCOUNT_JSON  # JSON string for a service account

Notes:
- The Google Doc prompt and input fields expect public/viewable docs for simple fetches.
- Google Docs creation uses the Docs API and inserts plain text into a newly created blank doc.
- Token totals are stored in a local SQLite file next to the app.
"""

import streamlit as st


def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("Login")

    username = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Log in"):
        if (
            username == st.secrets["APP_USERNAME"]
            and password == st.secrets["APP_PASSWORD"]
        ):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid login or password")

    return False


if not check_password():
    st.stop()

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "token_usage.sqlite3"

ADVANCED_PROMPT_SOURCE = (
    "https://docs.google.com/document/d/1fqFrF3_mUo7MZfKQktt7345GqSZqe0VcKZpXsBtLPmc/edit?tab=t.0"
)

LANGUAGES = ["Russian"]

MODEL_CHOICES = {
    "Claude Sonnet": {"provider": "anthropic", "model": "claude-sonnet-4.6"},
    "Claude Opus": {"provider": "anthropic", "model": "claude-opus-4.8"},
    "GPT": {"provider": "openai", "model": "gpt-5.5"},
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
    """Rough fallback when provider usage is unavailable."""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        # Cheap fallback: good enough for a rough estimate.
        return max(1, len(text.split()))


def extract_google_doc_id(url: str) -> Optional[str]:
    if not url:
        return None
    match = re.search(r"/document/d/([a-zA-Z0-9-_]+)", url)
    return match.group(1) if match else None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_public_google_doc_text(url: str) -> str:
    """Fetch plain text from a public Google Doc."""
    doc_id = extract_google_doc_id(url)
    if not doc_id:
        raise ValueError("Could not find a Google Doc ID in the link.")

    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    response = requests.get(export_url, timeout=30)
    response.raise_for_status()
    return response.text.strip()


def resolve_prompt(prompt_mode: str, language: str, prompt_doc_url: str, custom_prompt: str) -> str:
    if prompt_mode == "Simple prompt":
        return f"Translate the following into {language}"

    if prompt_mode == "Advanced prompt":
        prompt_text = fetch_public_google_doc_text(ADVANCED_PROMPT_SOURCE)
        return prompt_text.replace("$LANGUAGE", language)

    if prompt_mode == "Google Doc":
        prompt_text = fetch_public_google_doc_text(prompt_doc_url)
        return prompt_text.replace("$LANGUAGE", language)

    if prompt_mode == "Custom prompt":
        return custom_prompt.strip().replace("$LANGUAGE", language)

    raise ValueError("Unknown prompt mode.")


def resolve_input(input_mode: str, pasted_text: str, input_doc_url: str) -> str:
    if input_mode == "Paste text":
        return pasted_text.strip()
    if input_mode == "Google Doc":
        return fetch_public_google_doc_text(input_doc_url)
    raise ValueError("Unknown input mode.")


def get_anthropic_client():
    from anthropic import Anthropic  # type: ignore

    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY.")
    return Anthropic(api_key=api_key)


def get_openai_client():
    from openai import OpenAI  # type: ignore

    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    return OpenAI(api_key=api_key)


def call_model(provider: str, model: str, prompt: str, user_text: str) -> Tuple[str, int, int]:
    """Return (output_text, input_tokens, output_tokens)."""
    if provider == "anthropic":
        client = get_anthropic_client()
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=prompt,
            messages=[{"role": "user", "content": user_text}],
        )
        output_text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        if not input_tokens:
            input_tokens = get_token_count_fallback(prompt + "\n" + user_text)
        if not output_tokens:
            output_tokens = get_token_count_fallback(output_text)
        return output_text.strip(), input_tokens, output_tokens

    if provider == "openai":
        client = get_openai_client()
        response = client.responses.create(
            model=model,
            instructions=prompt,
            input=user_text,
            max_output_tokens=1024,
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            # Best-effort fallback if SDK shape changes.
            output_parts = []
            for item in getattr(response, "output", []) or []:
                for content in getattr(item, "content", []) or []:
                    text = getattr(content, "text", None)
                    if text:
                        output_parts.append(text)
            output_text = "".join(output_parts)

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        if not input_tokens:
            input_tokens = get_token_count_fallback(prompt + "\n" + user_text)
        if not output_tokens:
            output_tokens = get_token_count_fallback(output_text)
        return output_text.strip(), input_tokens, output_tokens

    raise ValueError(f"Unsupported provider: {provider}")


@st.cache_data(ttl=300, show_spinner=False)
def create_google_doc_with_text(title: str, text: str) -> str:
    """Create a Google Doc and insert plain text into it.

    Requires a service account JSON in GOOGLE_SERVICE_ACCOUNT_JSON.
    """
    service_json = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not service_json:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON.")

    from google.oauth2 import service_account  # type: ignore
    from googleapiclient.discovery import build  # type: ignore

    info = json.loads(service_json)
    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/drive.file",
        ],
    )

    docs_service = build("docs", "v1", credentials=creds, cache_discovery=False)
    created = docs_service.documents().create(body={"title": title}).execute()
    document_id = created["documentId"]

    # Insert text at the beginning of the blank document.
    docs_service.documents().batchUpdate(
        documentId=document_id,
        body={
            "requests": [
                {
                    "insertText": {
                        "location": {"index": 1},
                        "text": text,
                    }
                }
            ]
        },
    ).execute()

    return f"https://docs.google.com/document/d/{document_id}/edit"


def first_n_words(text: str, n: int = 200) -> str:
    words = text.split()
    if len(words) <= n:
        return text.strip()
    return " ".join(words[:n]).strip() + " …"


st.set_page_config(page_title="Translation App", layout="wide")
init_db()

st.title("Translation Stub")
st.caption("Language is fixed to Russian for now; the rest is wired for future expansion.")

top_left, top_right = st.columns([1, 1])
with top_left:
    language = st.selectbox("Select language", LANGUAGES, index=0)
with top_right:
    model_label = st.selectbox("Select model", list(MODEL_CHOICES.keys()), index=0)

model_info = MODEL_CHOICES[model_label]

st.divider()

prompt_mode = st.radio(
    "Prompt selection",
    ["Simple prompt", "Advanced prompt", "Google Doc", "Custom prompt"],
    horizontal=True,
)

prompt_doc_url = ""
custom_prompt = ""

if prompt_mode == "Google Doc":
    prompt_doc_url = st.text_input(
        "Google Doc link for the prompt",
        placeholder="https://docs.google.com/document/d/.../edit",
        help="Make sure the Google Doc is set to View for everyone.",
    )
elif prompt_mode == "Custom prompt":
    custom_prompt = st.text_area(
        "Custom prompt",
        height=180,
        placeholder="Type or paste your prompt here.",
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
else:
    input_doc_url = st.text_input(
        "Google Doc link for input text",
        placeholder="https://docs.google.com/document/d/.../edit",
        help="Make sure the Google Doc is set to View for everyone.",
    )

translate_clicked = st.button("Translate", type="primary", use_container_width=False)

if translate_clicked:
    try:
        prompt_text = resolve_prompt(prompt_mode, language, prompt_doc_url, custom_prompt)
        input_text = resolve_input(input_mode, pasted_text, input_doc_url)

        if not input_text:
            st.error("Input text is empty.")
        else:
            with st.spinner("Translating..."):
                output_text, input_tokens, output_tokens = call_model(
                    model_info["provider"],
                    model_info["model"],
                    prompt_text,
                    input_text,
                )

            total_tokens = save_operation(
                model_info["provider"],
                model_info["model"],
                input_tokens,
                output_tokens,
            )
            all_time_tokens = get_total_tokens_all_time()

            st.session_state["last_output"] = output_text
            st.session_state["last_prompt"] = prompt_text
            st.session_state["last_input"] = input_text
            st.session_state["last_input_tokens"] = input_tokens
            st.session_state["last_output_tokens"] = output_tokens
            st.session_state["last_total_tokens"] = total_tokens
            st.session_state["last_all_time_tokens"] = all_time_tokens
            st.session_state["last_model_label"] = model_label
            st.session_state["last_model_name"] = model_info["model"]
            st.session_state["last_provider"] = model_info["provider"]
    except Exception as exc:
        st.error(str(exc))

if "last_output" in st.session_state:
    st.divider()
    st.subheader("Preview")

    preview_text = first_n_words(st.session_state["last_output"], 200)
    st.code(preview_text, language=None)

    col_copy, col_doc = st.columns([1, 1])
    with col_copy:
        st.download_button(
            "Copy as .txt",
            data=st.session_state["last_output"].encode("utf-8"),
            file_name="translation.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_doc:
        doc_title = st.text_input(
            "Google Doc title",
            value=f"Translation {datetime.now().strftime('%Y-%m-%d %H-%M')}",
            key="doc_title",
        )
        if st.button("Create Google Doc with output", use_container_width=True):
            try:
                doc_url = create_google_doc_with_text(doc_title, st.session_state["last_output"])
                st.success("Google Doc created.")
                st.link_button("Open document", doc_url)
            except Exception as exc:
                st.error(str(exc))

    st.caption(
        f"This run used about {st.session_state['last_input_tokens'] + st.session_state['last_output_tokens']} tokens "
        f"({st.session_state['last_input_tokens']} input + {st.session_state['last_output_tokens']} output)."
    )
    st.caption(f"All-time token total stored locally: {st.session_state['last_all_time_tokens']}.")

else:
    st.caption("No translation has been run yet.")

st.divider()
st.caption(
    ""
    ""
)
