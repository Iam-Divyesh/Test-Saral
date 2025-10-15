# file: openai_clients_and_helpers.py

from openai import AzureOpenAI
import os
import json
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

# ---------- CONFIG ----------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Use different API-version strings when appropriate
AZURE_OPENAI_API_VERSION_EMBEDDING = os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING", "2024-02-01")
AZURE_OPENAI_API_VERSION_CHAT = os.getenv("AZURE_OPENAI_API_VERSION_CHAT", "2024-12-01-preview")

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
    st.error("Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT in environment variables.")
    # Depending on your app flow, you may want to raise or stop execution here
    # raise RuntimeError("Missing Azure OpenAI configuration")


# ---------- CLIENT INITIALIZATION ----------
def setup_openai_clients():
    """Create separate AzureOpenAI clients for embeddings and chat (different api versions)."""
    embedding_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION_EMBEDDING,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

    chat_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION_CHAT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

    return embedding_client, chat_client

embedding_client, chat_client = setup_openai_clients()


# ---------- GPT-ENHANCED QUERY PARSING ----------
def enhance_query_with_gpt(user_query: str) -> dict:
    """Enhanced query parsing with GPT-5-mini - returns complete JSON following rules."""
    system_prompt = """Extract key information from a recruitment query and return JSON:
{
    "job_title": "job title",
    "skills": ["skill1", "skill2"],
    "experience": 2,
    "location": ["exact locations only"]
}

RULES:
- For location: ONLY include exact locations mentioned (no states or countries).
- For experience:
    * If a range is mentioned (e.g., "2-4 years", "3 to 6 years", "4–5 yrs"), extract the MINIMUM numeric value.
    * If a single value is mentioned (e.g., "3 years"), return that number only.
    * If "fresher" or "entry level" mentioned, return 0.
    * Return only numeric integer values, no text or units.
- Keep responses concise.
- Return valid JSON only.
"""

    try:
        response = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze: '{user_query}'"}
            ],
            max_completion_tokens=800,
            reasoning_effort="low"
        )

        content = response.choices[0].message.content
        if content:
            # Safely parse JSON — allow for trailing whitespace/newlines
            try:
                parsed = json.loads(content.strip())
                # Minimal validation of expected keys
                if isinstance(parsed, dict) and "job_title" in parsed:
                    return parsed
            except json.JSONDecodeError:
                st.warning("enhance_query_with_gpt: GPT returned invalid JSON; falling back to heuristic.")
    except Exception as e:
        st.warning(f"Query enhancement error: {e}")

    # Fallback: best-effort simple heuristic
    return {
        "job_title": user_query,
        "skills": [],
        "experience": 0,
        "location": []
    }


# ---------- EMBEDDING HELPERS (with caching) ----------
_embedding_cache = {}

@lru_cache(maxsize=1000)
def _get_cached_embedding(query_text: str):
    """Internal cached embedding function. Returns embedding vector or None on error."""
    try:
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query_text]
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None


def get_query_embedding(query_text: str):
    """Generate embedding for query with caching"""
    if not query_text:
        return None
    return _get_cached_embedding(query_text)


def get_batch_embeddings(texts: list):
    """Generate embeddings for multiple texts in one API call"""
    if not texts:
        return []
    try:
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Batch embedding error: {e}")
        return [None] * len(texts)


# ---------- SIMILARITY & SEARCH ----------
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity safely (returns 0.0 if invalid)."""
    try:
        a = np.array(vec1, dtype=float)
        b = np.array(vec2, dtype=float)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    except Exception:
        return 0.0


def perform_semantic_search(candidates, query_embedding):
    """Calculate similarity scores for candidates using vectorized operations"""
    if not query_embedding or not candidates:
        return candidates

    valid_candidates = []
    embeddings_list = []

    for candidate in candidates:
        try:
            candidate_embedding = candidate.get('embedding')
            if not candidate_embedding:
                continue

            # If stored as JSON string, parse it
            if isinstance(candidate_embedding, str):
                candidate_embedding = json.loads(candidate_embedding)

            # Skip if still invalid
            if not candidate_embedding:
                continue

            valid_candidates.append(candidate)
            embeddings_list.append(candidate_embedding)
        except Exception:
            continue

    if not embeddings_list:
        return []

    query_arr = np.array(query_embedding, dtype=float)
    candidates_arr = np.array(embeddings_list, dtype=float)

    # Prevent division by zero; compute norms per row
    query_norm = np.linalg.norm(query_arr)
    candidates_norms = np.linalg.norm(candidates_arr, axis=1)
    denom = candidates_norms * query_norm
    # Avoid divide-by-zero
    denom[denom == 0] = 1e-12

    similarities = np.dot(candidates_arr, query_arr) / denom

    for idx, candidate in enumerate(valid_candidates):
        similarity = float(similarities[idx])
        candidate['similarity_score'] = similarity
        candidate['match_percentage'] = round(similarity * 100, 2)

    valid_candidates.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
    return valid_candidates


# ---------- GPT FILTERING ----------
def filter_by_gpt_relevance(candidates, target_role):
    """Filter candidates using GPT-5-mini relevance check"""
    if not target_role or not candidates:
        return candidates

    relevant = []
    progress_bar = st.progress(0)
    status = st.empty()

    for i, candidate in enumerate(candidates):
        try:
            progress_bar.progress((i + 1) / len(candidates))
            status.text(f"Validating {i + 1}/{len(candidates)}")

            headline = candidate.get('headline', 'No headline')

            response = chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": 'Return JSON: {"status": "RELEVANT" or "IRRELEVANT", "score": 85}'},
                    {"role": "user", "content": f"Role: {target_role}\nCandidate: {headline}"}
                ],
                max_completion_tokens=500,
                reasoning_effort="minimal"
            )

            content = response.choices[0].message.content
            if content:
                try:
                    result = json.loads(content.strip())
                    if result.get("status", "").upper() == "RELEVANT":
                        candidate['gpt_relevance'] = True
                        candidate['gpt_score'] = result.get("score", 0)
                        relevant.append(candidate)
                except json.JSONDecodeError:
                    # If GPT fails to return JSON, treat as relevant (fail-open)
                    candidate['gpt_relevance'] = True
                    relevant.append(candidate)
        except Exception:
            # On any error, include candidate (fail-open)
            candidate['gpt_relevance'] = True
            relevant.append(candidate)

    progress_bar.empty()
    status.empty()
    return relevant


# ---------- GOOGLE DORK BUILDER ----------
def gpt_build_dork(chat_client, model: str, user_query: str) -> str:
    """
    Ask GPT to convert a free-text recruitment prompt into a single Google dork string.
    - Uses AzureOpenAI chat completions API shape (chat_client.chat.completions.create).
    - Returns a single-line string that MUST start with "site:linkedin.com/in ..."
    - If GPT doesn't return a valid dork, returns an alert string.
    NOTE: This function intentionally contains NO try/except blocks to surface errors.
    """
    system_prompt = """
You are a recruitment assistant. The user will provide a short recruitment prompt.
Return ONLY the Google dork string (plain text) — no JSON, no explanation, nothing else.
The dork MUST:
- Begin with: site:linkedin.com/in
- Include the job title as a quoted phrase (and also include the gerund form if appropriate, e.g., "video editor" OR "video editing")
- Include experience phrases for integer years, e.g., "2 years" OR "2 yrs" OR "2+ years" OR "2 years experience" OR "2 yrs experience"
- Include common location variants when a city is given, e.g., "Surat" OR "Surat, Gujarat" OR "Surat Area"
- Always append: -site:linkedin.com/jobs -intitle:(jobs OR hiring OR vacancy OR vacancies OR career OR apply)
Do not output any other text.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    response = chat_client.chat.completions.create(
        model=model,
        messages=messages
    )

    content = response.choices[0].message.content.strip()

    if content.lower().startswith("site:linkedin.com/in"):
        return content
    else:
        return "ALERT: gpt-5-mini not working. Please check model or API response."
