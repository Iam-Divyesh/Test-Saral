
import re
import json
import asyncio
from sentence_transformers import SentenceTransformer, util

# Global model cache
_model_instance = None
_util = None

# Precompile regex patterns for performance
DURATION_PATTERN = re.compile(r'·\s*(.+)')
YEAR_PATTERN = re.compile(r'(\d+)\s*yr')
MONTH_PATTERN = re.compile(r'(\d+)\s*mo')

def get_model():
    """Get cached transformer model - loads once per process"""
    global _model_instance, _util
    if _model_instance is None:
        from sentence_transformers import SentenceTransformer, util as st_util
        _model_instance = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _util = st_util
    return _model_instance, _util

# Preload model at module import
get_model()

def extract_duration_from_caption(caption):
    """Extract duration text from caption like 'Aug 2023 - Present · 2 yrs 1 mo'."""
    if not caption:
        return None
    match = DURATION_PATTERN.search(caption)
    return match.group(1).strip() if match else None

def parse_duration_to_months(duration_str):
    """Convert text like '2 yrs 1 mo' or '3 mos' into total months."""
    if not duration_str:
        return 0
    years = months = 0
    year_match = YEAR_PATTERN.search(duration_str)
    month_match = MONTH_PATTERN.search(duration_str)
    if year_match:
        years = int(year_match.group(1))
    if month_match:
        months = int(month_match.group(1))
    return years * 12 + months

def calculate_total_experience(experiences):
    """Sum all experience durations from array."""
    total_months = 0
    for exp in experiences:
        sub_list = exp.get("subComponents", [])
        if exp.get("breakdown") and isinstance(sub_list, list) and len(sub_list) > 1:
            for sub in sub_list:
                caption = sub.get("caption", "")
                duration_text = extract_duration_from_caption(caption)
                total_months += parse_duration_to_months(duration_text)
        else:
            caption = exp.get("caption", "")
            duration_text = extract_duration_from_caption(caption)
            total_months += parse_duration_to_months(duration_text)
    return total_months

def check_role_match(experiences, job_role, similarity_threshold=0.65):
    """Check if candidate has role experience using both keyword and transformer similarity."""
    if not job_role:
        return False, 0

    job_role_lower = job_role.lower().strip()
    matched = False
    role_months = 0

    # Collect all titles first
    title_data = []
    for exp in experiences:
        sub_list = exp.get("subComponents", [])
        if exp.get("breakdown") and isinstance(sub_list, list) and len(sub_list) > 1:
            for sub in sub_list:
                title = sub.get("title", "")
                caption = sub.get("caption", "")
                if title:
                    title_data.append((title, caption))
        else:
            title = exp.get("title", "")
            caption = exp.get("caption", "")
            if title:
                title_data.append((title, caption))

    if not title_data:
        return False, 0

    titles_only = [t[0].lower() for t in title_data]

    # Quick keyword check first
    for i, (title, caption) in enumerate(title_data):
        if job_role_lower in titles_only[i]:
            matched = True
            duration_text = extract_duration_from_caption(caption)
            role_months += parse_duration_to_months(duration_text)

    # Only use transformer if no keyword match
    if not matched and titles_only:
        model, util = get_model()
        role_emb = model.encode(job_role_lower, convert_to_tensor=True, show_progress_bar=False)
        title_embs = model.encode(titles_only, convert_to_tensor=True, show_progress_bar=False)
        similarities = util.cos_sim(role_emb, title_embs)[0]

        for i, sim in enumerate(similarities):
            if sim.item() >= similarity_threshold:
                matched = True
                caption = title_data[i][1]
                duration_text = extract_duration_from_caption(caption)
                role_months += parse_duration_to_months(duration_text)

    return matched, role_months

def extract_experience_years(exp_value):
    """Extract integer years from value (already integer or string)."""
    if exp_value is None:
        return 0
    if isinstance(exp_value, int):
        return exp_value
    if isinstance(exp_value, str):
        match = re.search(r'(\d+)', exp_value)
        return int(match.group(1)) if match else 0
    return 0

def normalize_skills(skills):
    """Handle skill formats: list of strings, list of dicts, or single string."""
    normalized = []
    if not skills:
        return normalized
    if isinstance(skills, str):
        normalized = [s.strip().lower() for s in re.split(r'[;,]', skills) if s.strip()]
    elif isinstance(skills, list):
        for s in skills:
            if isinstance(s, dict):
                val = s.get("name") or s.get("skill") or s.get("title") or ""
                if val:
                    normalized.append(val.lower())
            elif isinstance(s, str):
                normalized.append(s.lower())
    return normalized

def calculate_skills_score(candidate, required_skills):
    """Safe skill matching across mixed input structures."""
    candidate_skills = (
        candidate.get("skills")
        or candidate.get("keySkills")
        or candidate.get("skillset")
        or []
    )
    candidate_skills = normalize_skills(candidate_skills)
    required_skills = normalize_skills(required_skills)
    if not required_skills:
        return 0
    matches = sum(1 for skill in required_skills if skill in candidate_skills)
    return round((matches / len(required_skills)) * 100, 1)

def calculate_headline_score(headline, job_keywords):
    """Score headline relevance."""
    if not headline or not job_keywords:
        return 0
    matches = sum(1 for kw in job_keywords if kw in headline.lower())
    return (matches / len(job_keywords)) * 100

def calculate_about_score(about, job_keywords):
    """Score about section relevance."""
    if not about or not job_keywords:
        return 0
    matches = sum(1 for kw in job_keywords if kw in about.lower())
    return (matches / len(job_keywords)) * 100

def calculate_experience_score_tiered(total_exp_years, role_exp_years, expected_exp_years, 
                                        skills_match_score=0, headline_match_score=0, about_match_score=0):
    """
    Tiered scoring based on new criteria:
    - First Priority (100 FP): Exact match or near match to expected experience
    - Second Priority (80-90): Total exp matches expected experience
    - Third Priority (<80): Both total and role are more than expected
    """
    if not expected_exp_years:
        keyword_score = (skills_match_score + headline_match_score + about_match_score) / 3
        return min(max(keyword_score, 40), 75)

    try:
        expected_exp_years = float(expected_exp_years)
    except:
        keyword_score = (skills_match_score + headline_match_score + about_match_score) / 3
        return min(max(keyword_score, 40), 75)

    # Convert years to months for precise comparison
    total_exp_months = total_exp_years * 12
    role_exp_months = role_exp_years * 12
    expected_exp_months = expected_exp_years * 12

    # Keyword average for fine-tuning
    keyword_avg = (skills_match_score + headline_match_score + about_match_score) / 3
    keyword_factor = keyword_avg / 100  # 0-1 scale

    # Calculate differences
    role_diff_months = abs(role_exp_months - expected_exp_months)
    total_diff_months = abs(total_exp_months - expected_exp_months)

    # --- FIRST PRIORITY: EXACT OR NEAR MATCH (100 FP) ---
    # Role experience matches expected (±6 months tolerance)
    if role_diff_months <= 6:
        base_score = 100
        # Small penalty if total exp doesn't also match
        if total_diff_months > 6:
            base_score -= min(total_diff_months / 12 * 2, 5)
        final_score = base_score + (keyword_factor * 0)  # Already at 100, no bonus needed
        return min(final_score, 100)

    # --- SECOND PRIORITY: TOTAL EXP MATCHES (80-90) ---
    # Total experience matches expected (±6 months tolerance)
    if total_diff_months <= 6:
        base_score = 85
        # Bonus if role exp is also close
        if role_diff_months <= 12:
            base_score += 3
        final_score = base_score + (keyword_factor * 2)
        return min(max(final_score, 80), 90)

    # --- THIRD PRIORITY: BOTH MORE THAN EXPECTED (<80) ---
    if role_exp_months > expected_exp_months and total_exp_months > expected_exp_months:
        re_excess_years = (role_exp_months - expected_exp_months) / 12
        te_excess_years = (total_exp_months - expected_exp_months) / 12
        avg_excess = (re_excess_years + te_excess_years) / 2

        base_score = 75

        # Penalty for being overqualified
        if avg_excess <= 2:
            penalty = avg_excess * 2
        elif avg_excess <= 4:
            penalty = 4 + (avg_excess - 2) * 3
        else:
            penalty = 10 + (avg_excess - 4) * 2

        final_score = base_score - penalty + (keyword_factor * 2)
        return min(max(final_score, 50), 79)

    # --- FOURTH PRIORITY: INSUFFICIENT EXPERIENCE (<50) ---
    else:
        keyword_base_score = keyword_avg * 0.5

        if expected_exp_years > 0:
            # Check how far role exp is from expected
            role_gap = max(0, expected_exp_months - role_exp_months) / 12

            if role_gap <= 0.5:
                proximity_score = 10
            elif role_gap <= 1:
                proximity_score = 7
            elif role_gap <= 1.5:
                proximity_score = 5
            elif role_gap <= 2:
                proximity_score = 3
            else:
                proximity_score = 0

            final_score = keyword_base_score + proximity_score
        else:
            final_score = keyword_base_score

        return min(max(final_score, 0), 49)

def get_score_tier(total_exp, role_exp, expected_exp):
    """Tier classification based on new scoring criteria"""
    if not expected_exp:
        return "Tier 4 (No Expected): 0-75"

    try:
        expected_exp = float(expected_exp)
    except:
        return "Tier 4 (Invalid Expected): 0-75"

    total_exp_months = total_exp * 12
    role_exp_months = role_exp * 12
    expected_exp_months = expected_exp * 12

    role_diff = abs(role_exp_months - expected_exp_months)
    total_diff = abs(total_exp_months - expected_exp_months)

    # First Priority: Role exp matches
    if role_diff <= 6:
        return "Tier 1 (Exact/Near Match): 95-100"
    
    # Second Priority: Total exp matches
    elif total_diff <= 6:
        return "Tier 2 (Total Exp Match): 80-90"
    
    # Third Priority: Both higher than expected
    elif role_exp_months > expected_exp_months and total_exp_months > expected_exp_months:
        return "Tier 3 (Overqualified): 50-79"
    
    # Fourth Priority: Insufficient
    else:
        return "Tier 4 (Insufficient): 0-49"

def calculate_sort_priority(candidate):
    """Enhanced sorting priority - score is primary."""
    score = candidate.get("score", 0)
    br = candidate.get("score_breakdown", {})
    
    ee = br.get("expected_experience_years", 0)
    re = br.get("role_experience_years", 0)
    te = br.get("total_experience_years", 0)

    re_diff = abs(re - ee) if ee else 0
    te_diff = abs(te - ee) if ee else 0

    # Primary: score (descending)
    # Secondary: role experience match (closer is better)
    # Tertiary: total experience match (closer is better)
    return (
        score,
        -(re_diff),
        -(te_diff)
    )

async def validate_candidate(candidate, location_filters, role, job_keywords, required_skills, expected_exp_years):
    """Async validation of single candidate."""
    # Location validation
    location = candidate.get("location") or candidate.get("addressWithCountry", "")
    if not location:
        return None

    location_lower = location.lower()
    location_valid = "india" in location_lower
    location_match = any(loc.lower() in location_lower for loc in location_filters)
    if not (location_valid and location_match):
        return None

    # Experience calculations
    experiences = candidate.get("experiences") or candidate.get("experience", [])
    if isinstance(experiences, str):
        try:
            experiences = json.loads(experiences)
        except:
            experiences = []

    total_exp_months = calculate_total_experience(experiences)
    total_exp_years = round(total_exp_months / 12, 2)

    role_matched, role_exp_months = check_role_match(experiences, role)
    role_exp_years = round(role_exp_months / 12, 2)

    if role and not role_matched:
        return None

    # Additional scores
    skills_score = calculate_skills_score(candidate, required_skills)
    headline_score = calculate_headline_score(candidate.get("headline", ""), job_keywords)
    about_score = calculate_about_score(candidate.get("about", ""), job_keywords)

    # Convert expected experience to float for scoring
    exp_years = float(expected_exp_years) if expected_exp_years else 0

    # Final score
    final_score = calculate_experience_score_tiered(
        total_exp_years,
        role_exp_years,
        exp_years,
        skills_score,
        headline_score,
        about_score,
    )

    # Assign scores
    candidate["score"] = round(final_score, 2)
    candidate["total_experience_years"] = total_exp_years
    candidate["role_experience_years"] = role_exp_years
    candidate["score_breakdown"] = {
        "total_experience_years": total_exp_years,
        "role_experience_years": role_exp_years,
        "expected_experience_years": exp_years,
        "skills_score": skills_score,
        "headline_score": headline_score,
        "about_score": about_score,
        "tier": get_score_tier(total_exp_years, role_exp_years, exp_years),
    }

    return candidate

async def validate_and_score_candidates_async(candidates, locations, role, experience_level, parsed_data):
    """Async validation with new tiered scoring system."""
    location_filters = [loc.lower().strip() for loc in locations] if locations else ["india"]

    job_keywords, required_skills = [], []
    if parsed_data:
        job_title = parsed_data.get("job_role", "") or parsed_data.get("job_title", "")
        if job_title:
            job_keywords = [kw.lower() for kw in job_title.split() if len(kw) > 2]
        skills = parsed_data.get("key_skills", []) or parsed_data.get("skills", [])
        required_skills = normalize_skills(skills)

    expected_exp_years = extract_experience_years(experience_level) if experience_level else 0

    # Process all candidates concurrently
    tasks = [
        validate_candidate(candidate, location_filters, role, job_keywords, required_skills, expected_exp_years)
        for candidate in candidates
    ]
    results = await asyncio.gather(*tasks)

    # Filter out None results
    validated = [r for r in results if r is not None]

    # Sort by priority
    validated.sort(key=calculate_sort_priority, reverse=True)

    return validated, []

def validate_and_score_candidates(candidates, locations, role, experience_level, parsed_data):
    """Sync wrapper for async validation."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            validate_and_score_candidates_async(candidates, locations, role, experience_level, parsed_data)
        )
    finally:
        loop.close()
