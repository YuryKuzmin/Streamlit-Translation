"""Streamlit translation stub app.

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
  APP_USERNAME
  APP_PASSWORD

Notes:
- The Google Doc prompt and input fields expect public/viewable docs for simple fetches.
- Token totals are stored in a local SQLite file next to the app.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
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


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "token_usage.sqlite3"

ADVANCED_PROMPT_SOURCE = (
    "https://docs.google.com/document/d/1fqFrF3_mUo7MZfKQktt7345GqSZqe0VcKZpXsBtLPmc/edit?tab=t.0"
)

LANGUAGES = ["Russian"]

MODEL_CHOICES = {
    "Claude Sonnet": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "Claude Opus": {"provider": "anthropic", "model": "claude-opus-4-8"},
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


def chunk_text(text: str, max_words: int = 10000) -> list[str]:
    """Splits text into chunks of roughly max_words, respecting paragraph breaks."""
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


def call_model(provider: str, model: str, prompt: str, user_text: str) -> Tuple[str, int, int]:
    MAX_OUTPUT_LIMIT = 30000

    if provider == "anthropic":
        client = get_anthropic_client()
        with client.messages.stream(
            model=model,
            max_tokens=MAX_OUTPUT_LIMIT,
            system=prompt,
            messages=[{"role": "user", "content": user_text}],
        ) as stream:
            response = stream.get_final_message()

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
            max_output_tokens=MAX_OUTPUT_LIMIT,
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
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
    
    if len(pasted_text.split()) > 3000:
        st.warning("You are about to translate a very large document.")
        
else:
    input_doc_url = st.text_input(
        "Google Doc link for input text",
        placeholder="https://docs.google.com/document/d/.../edit",
        help="Make sure the Google Doc is set to View for everyone.",
    )

# Manage translation state so button can be grayed out
if "is_translating" not in st.session_state:
    st.session_state.is_translating = False

def start_translation():
    st.session_state.is_translating = True

st.button(
    "Translate", 
    type="primary", 
    use_container_width=False, 
    disabled=st.session_state.is_translating,
    on_click=start_translation
)

# Run translation logic if button activated
if st.session_state.is_translating:
    try:
        prompt_text = resolve_prompt(prompt_mode, language, prompt_doc_url, custom_prompt)
        input_text = resolve_input(input_mode, pasted_text, input_doc_url)

        if not input_text:
            st.error("Input text is empty.")
        else:
            if len(input_text.split()) > 3000 and input_mode == "Google Doc":
                st.warning("You are about to translate a very large document.")

            chunks = chunk_text(input_text, max_words=10000)
            
            full_output_text = ""
            total_input_tokens = 0
            total_output_tokens = 0

            # ---------------------------------------------------------
            # PROGRESS BAR & ETA ESTIMATION
            # ---------------------------------------------------------
            # Rough proxy variables for estimated processing speeds (tokens/sec)
            model_speeds = {
                "claude-sonnet-4-6": 45.0,
                "claude-opus-4-8": 15.0,
                "gpt-5.5": 50.0
            }
            avg_speed = model_speeds.get(model_info["model"], 30.0)
            
            est_words = len(input_text.split())
            # Rough math: Words * 1.3 (Input Tokens) + Words * 1.5 (Output tokens translation expansion)
            est_total_tokens = est_words * 2.8 
            est_total_seconds = est_total_tokens / avg_speed

            progress_text = f"Translating document (0/{len(chunks)} chunks)... Estimated time: ~{int(est_total_seconds)}s"
            progress_bar = st.progress(0, text=progress_text)

            start_time = time.time()

            for i, chunk in enumerate(chunks):
                # Update ETA dynamically based on chunks left
                elapsed = time.time() - start_time
                chunks_left = len(chunks) - i
                
                if i > 0:
                    avg_time_per_chunk = elapsed / i
                    current_eta = int(avg_time_per_chunk * chunks_left)
                else:
                    current_eta = int(est_total_seconds)

                progress_bar.progress(i / len(chunks), text=f"Translating chunk {i+1} of {len(chunks)}... (ETA: ~{current_eta}s)")
                
                chunk_output, in_tokens, out_tokens = call_model(
                    model_info["provider"],
                    model_info["model"],
                    prompt_text,
                    chunk,
                )
                
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
            st.session_state["last_model_name"] = model_info["model"]
            st.session_state["last_provider"] = model_info["provider"]
            
    except Exception as exc:
        st.error(str(exc))
    finally:
        # Reset translating state and refresh UI to re-enable button
        st.session_state.is_translating = False
        st.rerun()

if "last_output" in st.session_state:
    st.divider()
    st.subheader("Output")
    
    st.caption("Use the 'Copy' icon in the top right corner of the block below to copy the translation.")
    
    # st.code natively provides a nice "copy to clipboard" button
    st.code(st.session_state["last_output"], language=None)

    col1, col2 = st.columns([1, 4])
    with col1:
        st.download_button(
            "Download as txt",
            data=st.session_state["last_output"].encode("utf-8"),
            file_name="translation.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.caption(
        f"This run used about {st.session_state['last_input_tokens'] + st.session_state['last_output_tokens']} tokens "
        f"({st.session_state['last_input_tokens']} input + {st.session_state['last_output_tokens']} output)."
    )
    st.caption(f"All-time token total stored locally: {st.session_state['last_all_time_tokens']}.")

else:
    st.caption("No translation has been run yet.")

st.divider()
st.caption("")
