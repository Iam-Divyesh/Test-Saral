
import streamlit as st
import json
import time
from postgres_v3 import get_data_with_embeddings, store_apify_profiles
from nlp_v3 import (
    setup_openai_clients,
    enhance_query_with_gpt,
    get_query_embedding,
    perform_semantic_search,
    filter_by_gpt_relevance
)
from serp_v3 import query_making, serp_api_call
from apify_v3 import apify_call
from validate_v3 import validate_and_score_candidates

# Initialize session state
if "v2_results" not in st.session_state:
    st.session_state.v2_results = []
if "v1_results" not in st.session_state:
    st.session_state.v1_results = []
if "final_results" not in st.session_state:
    st.session_state.final_results = []

st.set_page_config(page_title="SARAL AI v3 Pro Max", page_icon="🚀", layout="wide")

st.header("🚀 SARAL AI v3 Pro Max")
st.subheader("Hybrid Search: Semantic DB + Live SERP Scraping")

user_input = st.text_area(
    "Enter Your Search Query",
    placeholder="e.g., UI Designer with 2-3 years experience in Surat",
    height=100
)

if st.button("🔍 Search Candidates", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter a valid query")
        st.stop()

    # Clear previous results
    st.session_state.v2_results = []
    st.session_state.v1_results = []
    st.session_state.final_results = []

    # Main progress container
    main_progress = st.progress(0)
    status_text = st.empty()

    # ========== PART 1: SEMANTIC SEARCH (V2 APPROACH) ==========
    st.write("---")
    st.subheader("📊 Part 1: Semantic Database Search")

    with st.spinner("Analyzing query with GPT..."):
        enhanced_query = enhance_query_with_gpt(user_input)
        query_embedding = get_query_embedding(user_input)
    
    main_progress.progress(0.10)
    status_text.text("Query enhanced and embedded")

    st.write("**Enhanced Query Analysis (JSON):**")
    st.json(enhanced_query)

    # Fetch candidates from database
    with st.spinner("Fetching candidates from database..."):
        db_candidates = get_data_with_embeddings()
    
    st.info(f"📦 Retrieved {len(db_candidates)} candidates from database")
    main_progress.progress(0.20)

    # Location filtering
    target_locations = enhanced_query.get("location", [])
    if target_locations:
        loc_filtered = []
        for candidate in db_candidates:
            location = candidate.get('location', '')
            if location:
                location_lower = str(location).lower()
                if any(str(loc).lower() in location_lower for loc in target_locations):
                    loc_filtered.append(candidate)
        st.write(f"After location filter: {len(loc_filtered)} candidates")
    else:
        loc_filtered = db_candidates

    main_progress.progress(0.30)

    # Semantic search WITHOUT GPT validation
    if query_embedding and loc_filtered:
        with st.spinner("Performing semantic search..."):
            semantic_results = perform_semantic_search(loc_filtered, query_embedding)
        
        main_progress.progress(0.40)
        st.write(f"Semantic search complete: {len(semantic_results)} candidates")

        # Validate using validate.py method instead of GPT
        with st.spinner("Validating candidates with validate.py..."):
            v2_candidates, _ = validate_and_score_candidates(
                semantic_results,
                enhanced_query.get("location", []),
                enhanced_query.get("job_role", ""),
                enhanced_query.get("experience_level", ""),
                enhanced_query
            )

        # Take top 10 from v2
        v2_top_10 = v2_candidates[:10]
        st.session_state.v2_results = v2_top_10
        st.success(f"✅ Part 1 Complete: {len(v2_top_10)} candidates from semantic search")
    else:
        st.warning("No semantic search results")
        v2_top_10 = []

    main_progress.progress(0.50)

    # ========== PART 2: SERP + APIFY SCRAPING (V1 APPROACH) ==========
    st.write("---")
    st.subheader("🌐 Part 2: Live SERP + LinkedIn Scraping")

    # Generate SERP query
    serp_query, _ = query_making(enhanced_query)
    st.code(serp_query, language="text")

    v1_candidates = []
    serp_page = 0
    max_serp_attempts = 5
    target_v1_candidates = 10

    with st.spinner("Scraping LinkedIn profiles..."):
        while len(v1_candidates) < target_v1_candidates and serp_page < max_serp_attempts:
            status_text.text(f"SERP page {serp_page + 1}, found {len(v1_candidates)}/10 candidates")
            
            start = serp_page * 10
            serp_data = serp_api_call(serp_query, start=start, results_per_page=10)

            if not serp_data or not serp_data.get('organic_results'):
                st.warning(f"No more SERP results at page {serp_page + 1}")
                break

            # Extract LinkedIn URLs
            all_linkedin_urls = {}
            for idx, result in enumerate(serp_data.get('organic_results', []), start=1):
                link = result.get('link', '')
                if 'linkedin.com/in/' in link:
                    clean_link = link.replace('in.linkedin.com', 'linkedin.com')
                    all_linkedin_urls[idx] = clean_link

            if not all_linkedin_urls:
                break

            # Check database first for existing profiles
            db_profiles = []
            urls_to_scrape = {}
            
            try:
                from postgres_v3 import get_connection
                conn = get_connection()
                cur = conn.cursor()
                
                urls_list = list(all_linkedin_urls.values())
                cur.execute("""
                    SELECT name, location, email, linkedin_url, headline, 
                           skills, about, experience, profile_pic
                    FROM profiles
                    WHERE linkedin_url = ANY(%s)
                """, (urls_list,))
                
                rows = cur.fetchall()
                found_urls = set()
                
                for row in rows:
                    found_urls.add(row[3])
                    db_profiles.append({
                        'fullName': row[0],
                        'addressWithCountry': row[1],
                        'email': row[2],
                        'linkedinUrl': row[3],
                        'headline': row[4],
                        'skills': row[5],
                        'about': row[6],
                        'experiences': row[7],
                        'profilePic': row[8]
                    })
                
                cur.close()
                conn.close()
                
                # Determine which URLs need scraping
                for idx, url in all_linkedin_urls.items():
                    if url not in found_urls:
                        urls_to_scrape[idx] = url
                
                st.info(f"📦 Found {len(db_profiles)} in database, {len(urls_to_scrape)} to scrape")
                
            except Exception as db_error:
                st.warning(f"Database check failed: {db_error}")
                urls_to_scrape = all_linkedin_urls

            # Scrape only new profiles with Apify
            apify_profiles = []
            if urls_to_scrape:
                try:
                    apify_profiles = apify_call(urls_to_scrape)
                except Exception as e:
                    st.error(f"Apify error: {e}")

            # Combine database and Apify results
            all_profiles = db_profiles + apify_profiles
            
            # Validate and score
            validated, _ = validate_and_score_candidates(
                all_profiles,
                enhanced_query.get("location", []),
                enhanced_query.get("job_role", ""),
                enhanced_query.get("experience_level", ""),
                enhanced_query
            )

            v1_candidates.extend(validated)
            
            # Store new profiles in DB for future use
            if apify_profiles:
                store_apify_profiles(apify_profiles)

            serp_page += 1
            main_progress.progress(0.50 + (serp_page / max_serp_attempts) * 0.30)

    # Take top 10 from v1
    v1_top_10 = sorted(v1_candidates, key=lambda x: x.get('score', 0), reverse=True)[:10]
    st.session_state.v1_results = v1_top_10
    st.success(f"✅ Part 2 Complete: {len(v1_top_10)} candidates from SERP scraping")

    main_progress.progress(0.80)

    # ========== PART 3: MERGE AND FINAL RANKING ==========
    st.write("---")
    st.subheader("🏆 Part 3: Merging and Final Ranking")

    # Combine results
    all_candidates = []
    seen_urls = set()

    # Add v2 results with source tag
    for candidate in v2_top_10:
        url = candidate.get('linkedin_url', '')
        if url and url not in seen_urls:
            candidate['source'] = 'Semantic Search'
            all_candidates.append(candidate)
            seen_urls.add(url)

    # Add v1 results with source tag
    for candidate in v1_top_10:
        url = candidate.get('linkedinUrl', '')
        if url and url not in seen_urls:
            candidate['source'] = 'SERP Scraping'
            all_candidates.append(candidate)
            seen_urls.add(url)

    # Re-score combined results
    final_candidates, _ = validate_and_score_candidates(
        all_candidates,
        enhanced_query.get("location", []),
        enhanced_query.get("job_role", ""),
        enhanced_query.get("experience_level", ""),
        enhanced_query
    )

    # Final top 10
    final_top_10 = final_candidates[:10]
    st.session_state.final_results = final_top_10

    main_progress.progress(1.0)
    status_text.text("✅ Search complete!")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Semantic Results", len(v2_top_10))
    with col2:
        st.metric("SERP Results", len(v1_top_10))
    with col3:
        st.metric("Total Unique", len(all_candidates))
    with col4:
        st.metric("Final Top", len(final_top_10))

    time.sleep(1)
    main_progress.empty()
    status_text.empty()

# ========== DISPLAY RESULTS ==========
if st.session_state.final_results:
    st.write("---")
    st.subheader("🎯 Final Top 10 Candidates")

    for idx, profiles in enumerate(st.session_state.final_results, start=1):
        # Handle both v1 and v2 field names
        name = profiles.get('name') or profiles.get('fullName', 'Unknown')
        
        # Get experience from score_breakdown
        score_breakdown = profiles.get('score_breakdown', {})
        total_exp_years = score_breakdown.get('total_experience_years', 0)
        exp_display = f"{total_exp_years} years" if total_exp_years else "N/A"

        with st.expander(
            f"{idx}. {name} • Experience: {exp_display}", expanded=True
        ):
            col1, col2 = st.columns([1, 2])

            # --- LEFT COLUMN ---
            with col1:
                temp_image = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRDVO09x_DXK3p4Mt1j08Ab0R875TdhsDcG2A&s"
                pic = profiles.get('profile_pic') or profiles.get('profilePic')
                st.image(pic if pic else temp_image, width=150)

                location = profiles.get('location') or profiles.get('addressWithCountry', '-')
                st.markdown(f"**Location:** {location}")
                st.markdown(f"**Email:** {profiles.get('email', 'None')}")

                # --- Determine Open to Work ---
                experiences = profiles.get('experiences', [])
                if isinstance(experiences, str):
                    try:
                        experiences = json.loads(experiences)
                    except:
                        experiences = []

                open_to_work = True
                for exp in experiences:
                    if isinstance(exp, dict):
                        caption = exp.get("caption", "")
                        if "Present" in caption:
                            open_to_work = False
                            break

                st.markdown(f"**Open to Work:** {'False' if not open_to_work else 'True'}")

                linkedin = profiles.get('linkedin_url') or profiles.get('linkedinUrl', '')
                if linkedin:
                    st.markdown(f"**LinkedIn:** [LinkedIn]({linkedin})")

            # --- RIGHT COLUMN ---
            with col2:
                st.markdown(f"### {name}")

                headline = profiles.get('headline')
                if headline:
                    st.markdown(f"*{headline}*")

                # --- Skills ---
                skills_raw = profiles.get('skills', [])
                if isinstance(skills_raw, str):
                    try:
                        skills_raw = json.loads(skills_raw)
                    except:
                        skills_raw = []

                skill_titles = [
                    s.get("title")
                    for s in skills_raw
                    if isinstance(s, dict) and "title" in s
                ]
                if skill_titles:
                    st.markdown("**Skills:** " + " • ".join(skill_titles[:10]))

                # --- About ---
                about = profiles.get('about')
                if about:
                    st.markdown(
                        "**About:** " + (about[:250] + "..." if len(about) > 250 else about)
                    )

                # --- Experience ---
                if experiences:
                    st.markdown("**Experience**")
                    for exp in experiences:
                        if isinstance(exp, dict):
                            title = exp.get("title", "")
                            subtitle = exp.get("subtitle") or exp.get("metadata", "")
                            caption = exp.get("caption", "")
                            st.write(f"• {title} at {subtitle} — {caption}")

                            # Description bullets (if any)
                            if exp.get("description"):
                                for desc in exp["description"]:
                                    if isinstance(desc, dict) and "text" in desc:
                                        st.markdown(f"    - {desc['text']}")

                # --- Profile Status ---
                if profiles.get('is_complete'):
                    st.markdown(f"**Profile Status:** {profiles.get('is_complete')}")

                # --- Score Breakdown (if available) ---
                if profiles.get('score_breakdown'):
                    with st.expander("Score Breakdown"):
                        st.json(profiles['score_breakdown'])


# Sidebar stats
with st.sidebar:
    st.subheader("Search Statistics")
    
    if st.session_state.final_results:
        st.metric("Final Candidates", len(st.session_state.final_results))
        
        # Source distribution
        semantic_count = sum(1 for c in st.session_state.final_results if c.get('source') == 'Semantic Search')
        serp_count = sum(1 for c in st.session_state.final_results if c.get('source') == 'SERP Scraping')
        
        st.write("**Source Distribution:**")
        st.write(f"• Semantic: {semantic_count}")
        st.write(f"• SERP: {serp_count}")
        
        # Average scores
        avg_score = sum(c.get('score', 0) for c in st.session_state.final_results) / len(st.session_state.final_results)
        st.metric("Average Score", f"{avg_score:.1f}")
