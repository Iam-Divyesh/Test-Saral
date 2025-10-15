
import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()

SERP_API_KEY = os.getenv("SERP_API_KEY","2ea94e751697921f9a04a148025f4dec7943956cb72ba83d7c95e15fe9c2a4db")


import re

def query_making(enhanced_query: dict):
    """Generate Google dork query for LinkedIn profiles (uses exact_experience)."""
    parts = ['site:linkedin.com/in']

    # Job role
    job_role = enhanced_query.get('job_role')
    if job_role:
        parts.append(f'( "{job_role}" )')

    # Skills (top 3)
    skills = enhanced_query.get('key_skills', []) or enhanced_query.get('skills', [])
    for s in skills[:3]:
        if s:
            parts.append(f'"{s}"')

    # Experience: prefer exact_experience, fall back to experience_level
    exp_raw = enhanced_query.get('exact_experience') or enhanced_query.get('experience_level') or ''
    exp_clause = None

    def parse_min_years(s: str):
        """Return integer minimum years, or special tokens 'FRESHER' or None."""
        if not s:
            return None
        s_low = s.lower().strip()
        if 'fresher' in s_low or 'entry' in s_low:
            return 'FRESHER'
        # look for range like '2-4', '2 to 4', '2–4'
        m_range = re.search(r'(\d+)\s*(?:-|to|–|and)\s*(\d+)', s_low)
        if m_range:
            try:
                return int(m_range.group(1))
            except:
                pass
        # look for single number possibly with + e.g. '3+ years' or '3 years'
        m_num = re.search(r'(\d+)\s*\+?', s_low)
        if m_num:
            try:
                return int(m_num.group(1))
            except:
                pass
        # if string contains words like 'senior'/'mid'/'junior', map roughly
        if 'junior' in s_low:
            return 1
        if 'mid' in s_low or 'mid-level' in s_low or 'mid level' in s_low:
            return 3
        if 'senior' in s_low:
            return 5
        return None

    min_years = parse_min_years(str(exp_raw))

    if min_years == 'FRESHER':
        exp_clause = '("Fresher")'
    elif isinstance(min_years, int):
        # Build clause that prioritizes minimal level
        if min_years <= 1:
            # include 1-2 and single years
            exp_clause = '( "1-2 years" OR "2 years" OR "1 year" )'
        elif min_years >= 5:
            # for senior and higher, use 5+ OR exact year
            exp_clause = f'( "5+ years" OR "{min_years} years" )'
        else:
            # for mid ranges, build "<min>-<min+2> years" then exact "<min> years"
            end = min_years + 2
            exp_clause = f'( "{min_years}-{end} years" OR "{min_years} years" )'
    else:
        # fallback: no experience filter
        exp_clause = None

    if exp_clause:
        parts.append(exp_clause)

    # Location
    locations = enhanced_query.get('location', []) or []
    if locations:
        loc_terms = []
        for loc in locations:
            if loc:
                loc_terms.append(f'"{loc}"')
                loc_terms.append(f'"{loc} Area"')
        if loc_terms:
            parts.append('( ' + ' OR '.join(loc_terms) + ' )')

    # Negative filters
    parts.append('-site:linkedin.com/jobs -intitle:("jobs" OR "hiring" OR "vacancy")')

    query = ' '.join(parts)
    return query, locations

 

def serp_api_call(query, start=0, results_per_page=10):
    """Call SERP API to get LinkedIn URLs"""
    params = {
        "engine": "google",
        "q": query.strip(),
        "api_key": SERP_API_KEY,
        "hl": "en",
        "gl": "in",
        "google_domain": "google.co.in",
        "location": "India",
        "num": results_per_page,
        "start": start,
        "safe": "active"
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"SERP API error: {e}")
    
    return None
