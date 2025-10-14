
from openai import AzureOpenAI
import os
import json
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# Initialize clients
def setup_openai_clients():
    embedding_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
        azure_endpoint="https://job-recruiting-bot.openai.azure.com/"
    )

    chat_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        azure_endpoint="https://job-recruiting-bot.openai.azure.com/"
    )

    return embedding_client, chat_client


embedding_client, chat_client = setup_openai_clients()

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"

def enhance_query_with_gpt(user_query: str) -> dict:
    """Enhanced query parsing with GPT-5-mini - returns complete JSON"""
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
- Return valid JSON only."""

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
            parsed = json.loads(content.strip())
            return parsed
    except Exception as e:
        st.warning(f"Query enhancement error: {e}")
    
    return {
        "job_title": user_query,
        "skills": [],
        "experience": 0,
        "location": []
    }


# Embedding cache to avoid redundant API calls (increased size)
from functools import lru_cache
_embedding_cache = {}

@lru_cache(maxsize=1000)
def _get_cached_embedding(query_text: str):
    """Internal cached embedding function"""
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
        return [data.embedding for data in response.data]
    except Exception as e:
        st.error(f"Batch embedding error: {e}")
        return [None] * len(texts)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity"""
    try:
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except:
        return 0.0

def perform_semantic_search(candidates, query_embedding):
    """Calculate similarity scores for candidates using vectorized operations"""
    if not query_embedding or not candidates:
        return candidates

    # Prepare arrays for vectorized computation
    valid_candidates = []
    embeddings_list = []
    
    for candidate in candidates:
        try:
            candidate_embedding = candidate.get('embedding')
            
            if candidate_embedding:
                if isinstance(candidate_embedding, str):
                    candidate_embedding = json.loads(candidate_embedding)
                
                valid_candidates.append(candidate)
                embeddings_list.append(candidate_embedding)
        except:
            continue
    
    if not embeddings_list:
        return []
    
    # Vectorized similarity computation
    query_arr = np.array(query_embedding)
    candidates_arr = np.array(embeddings_list)
    
    # Compute all similarities at once
    norms = np.linalg.norm(candidates_arr, axis=1) * np.linalg.norm(query_arr)
    similarities = np.dot(candidates_arr, query_arr) / norms
    
    # Assign scores
    for idx, candidate in enumerate(valid_candidates):
        similarity = float(similarities[idx])
        candidate['similarity_score'] = similarity
        candidate['match_percentage'] = round(similarity * 100, 2)
    
    # Sort by similarity
    valid_candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
    return valid_candidates

def filter_by_gpt_relevance(candidates, target_role):
    """Filter candidates using GPT-5-mini relevance check"""
    if not target_role or not candidates:
        return candidates

    relevant = []
    
    progress_bar = st.progress(0)
    status = st.empty()

    for i, candidate in enumerate(candidates):
        progress_bar.progress((i + 1) / len(candidates))
        status.text(f"Validating {i + 1}/{len(candidates)}")

        headline = candidate.get('headline', 'No headline')
        
        try:
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
                result = json.loads(content.strip())
                if result.get("status", "").upper() == "RELEVANT":
                    candidate['gpt_relevance'] = True
                    candidate['gpt_score'] = result.get("score", 0)
                    relevant.append(candidate)
        except:
            # On error, include candidate
            candidate['gpt_relevance'] = True
            relevant.append(candidate)

    progress_bar.empty()
    status.empty()

    return relevant

def gpt_build_dork(chat_client, model: str, user_query: str) -> str:
    """
    Ask GPT to convert a free-text recruitment prompt into a single Google dork string.
    - Uses AzureOpenAI chat completions API shape (chat_client.chat.completions.create).
    - Returns a single-line string that MUST start with "site:linkedin.com/in ..."
    - If GPT doesn't return a valid dork, returns an alert string.
    NOTE: This function intentionally contains NO try/except blocks.
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

    # Use the same AzureOpenAI call style you used in enhance_query_with_gpt
    response = chat_client.chat.completions.create(
        model=model,
        messages=messages
    )

    # Extract text from the SDK response (same pattern used elsewhere in your code)
    content = response.choices[0].message.content.strip()

    # Validate minimal requirement and return alert if not valid
    if content.lower().startswith("site:linkedin.com/in"):
        return content
    else:
        return "ALERT: gpt-5-mini not working. Please check model or API response."
