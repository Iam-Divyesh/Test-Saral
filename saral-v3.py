import streamlit as st
import json
import time
import asyncio
from postgres_v3 import get_data_with_embeddings, store_apify_profiles
from nlp_v3 import (
    enhance_query_with_gpt,
    get_query_embedding,
    perform_semantic_search,
    gpt_build_dork,
    chat_client,
    embedding_client
)
from serp_v3 import query_making, serp_api_call
from apify_v3 import apify_call
from validate_v3 import validate_and_score_candidates, get_model

@st.cache_resource
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = 'gpt-5-mini'

# Initialize session state
if "semantic_results" not in st.session_state:
    st.session_state.semantic_results = []
if "serp_results" not in st.session_state:
    st.session_state.serp_results = []
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
    st.session_state.semantic_results = []
    st.session_state.serp_results = []
    st.session_state.final_results = []

    # Main progress container
    main_progress = st.progress(0)
    status_text = st.empty()

    # ========== STEP 1: SEMANTIC SEARCH ON DB ==========
    st.write("---")
    st.subheader("📊 Step 1: Semantic Database Search")

    with st.spinner("Analyzing query with GPT..."):
        enhanced_query = enhance_query_with_gpt(user_input)

    try:
        query_embedding = get_query_embedding(user_input)
        if query_embedding:
            st.success(f"✅ Embedding created successfully (length: {len(query_embedding)})")
        else:
            st.error("❌ No embedding returned — check your embedding model name or Azure deployment.")
    except Exception as e:
        st.error(f"❌ Embedding creation failed: {e}")
        query_embedding = None

    main_progress.progress(0.10)
    status_text.text("Query enhanced and embedded")

    st.write("**Enhanced Query Analysis:**")
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

    # Semantic search
    semantic_candidates = []
    if query_embedding and loc_filtered:
        with st.spinner("Performing semantic search..."):
            semantic_results = perform_semantic_search(loc_filtered, query_embedding)

        main_progress.progress(0.40)
        st.write(f"Semantic search complete: {len(semantic_results)} candidates")

        # Limit to first 100 candidates for validation (speed optimization)
        candidates_to_validate = semantic_results[:200]
        st.info(f"Validating top {len(candidates_to_validate)} candidates for speed optimization")

        # Validate and score using async validate_v3.py
        with st.spinner("Validating and scoring with Sentence Transformer (async)..."):
            from validate_v3 import validate_and_score_candidates_async

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            semantic_candidates, _ = loop.run_until_complete(
                validate_and_score_candidates_async(
                    candidates_to_validate,
                    enhanced_query.get("location", []),
                    enhanced_query.get("job_title", ""),
                    enhanced_query.get("experience", 0),
                    enhanced_query
                )
            )
            loop.close()

        # Take top 10 from semantic search
        semantic_top_10 = semantic_candidates[:10]
        st.session_state.semantic_results = semantic_top_10
        st.success(f"✅ Step 1 Complete: {len(semantic_top_10)} candidates from semantic search")
    else:
        st.warning("No semantic search results")
        semantic_top_10 = []

    main_progress.progress(0.50)

    # ========== STEP 2: SERP + APIFY SCRAPING (ASYNC BATCH) ==========
    st.write("---")
    st.subheader("🌐 Step 2: Live SERP + LinkedIn Scraping (Async Batch)")

    # Generate SERP query
    serp_query = gpt_build_dork(chat_client, model, user_input)
    st.code(serp_query, language="text")

    serp_candidates = []
    max_serp_pages = 3  # Process 3 SERP pages
    target_serp_candidates = 10

    with st.spinner("Fetching SERP results from 3 pages in parallel..."):
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        # Fetch all 3 SERP pages in parallel
        async def fetch_serp_page(page_num):
            start = page_num * 10
            return serp_api_call(serp_query, start=start, results_per_page=10)

        async def fetch_all_serp_pages():
            tasks = [fetch_serp_page(i) for i in range(max_serp_pages)]
            return await asyncio.gather(*tasks)

        # Run async SERP fetching
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        serp_pages_data = loop.run_until_complete(fetch_all_serp_pages())
        loop.close()

        # Collect all LinkedIn URLs from all pages
        all_linkedin_urls_combined = {}
        url_counter = 1

        for serp_data in serp_pages_data:
            if serp_data and serp_data.get('organic_results'):
                for result in serp_data.get('organic_results', []):
                    link = result.get('link', '')
                    if 'linkedin.com/in/' in link:
                        clean_link = link

                        # Remove localhost prefix if present
                        if 'localhost' in clean_link:
                            if 'linkedin.com/in/' in clean_link:
                                parts = clean_link.split('linkedin.com/in/')
                                if len(parts) > 1:
                                    clean_link = f"https://www.linkedin.com/in/{parts[-1]}"

                        # Ensure https protocol
                        if not clean_link.startswith('http'):
                            clean_link = f"https://{clean_link}"

                        # Normalize
                        clean_link = clean_link.replace('in.linkedin.com', 'linkedin.com')

                        if 'linkedin.com/in/' in clean_link and clean_link not in all_linkedin_urls_combined.values():
                            all_linkedin_urls_combined[url_counter] = clean_link
                            url_counter += 1

        st.info(f"📋 Collected {len(all_linkedin_urls_combined)} unique LinkedIn URLs from 3 SERP pages")
        main_progress.progress(0.55)

        if not all_linkedin_urls_combined:
            st.warning("No LinkedIn URLs found in SERP results")
        else:
            # Check database for existing profiles
            db_profiles = []
            urls_to_scrape = {}

            try:
                from postgres_v3 import get_connection, return_connection
                conn = get_connection()
                cur = conn.cursor()

                urls_list = list(all_linkedin_urls_combined.values())
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
                return_connection(conn)

                # Determine which URLs need scraping
                for idx, url in all_linkedin_urls_combined.items():
                    if url not in found_urls:
                        urls_to_scrape[idx] = url

                st.info(f"📦 Found {len(db_profiles)} in DB, {len(urls_to_scrape)} to scrape")

            except Exception as db_error:
                st.warning(f"Database check failed: {db_error}")
                urls_to_scrape = all_linkedin_urls_combined

            main_progress.progress(0.60)

            # Batch scrape with Apify (10 URLs at a time, async)
            apify_profiles = []
            if urls_to_scrape:
                status_text.text(f"Scraping {len(urls_to_scrape)} URLs with Apify in batches of 10...")

                try:
                    # Take first 10 URLs for scraping to get results faster
                    batch_urls = dict(list(urls_to_scrape.items())[:10])
                    apify_profiles = apify_call(batch_urls)
                    st.success(f"✅ Scraped {len(apify_profiles)} profiles from Apify")
                except Exception as e:
                    st.error(f"Apify error: {e}")

            main_progress.progress(0.70)

            # Combine all profiles
            all_profiles = db_profiles + apify_profiles

            if all_profiles:
                # Validate and score all candidates asynchronously
                status_text.text("Validating and scoring candidates (async)...")
                from validate_v3 import validate_and_score_candidates_async

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                validated, _ = loop.run_until_complete(
                    validate_and_score_candidates_async(
                        all_profiles,
                        enhanced_query.get("location", []),
                        enhanced_query.get("job_title", ""),
                        enhanced_query.get("experience", 0),
                        enhanced_query
                    )
                )
                loop.close()

                serp_candidates.extend(validated)

                # Store new profiles
                if apify_profiles:
                    store_apify_profiles(apify_profiles)

    # Ensure exactly 10 candidates from SERP
    serp_top_10 = sorted(serp_candidates, key=lambda x: x.get('score', 0), reverse=True)[:10]

    # If less than 10, pad the message accordingly
    if len(serp_top_10) < 10:
        st.warning(f"Only {len(serp_top_10)} SERP candidates found (target: 10)")

    st.session_state.serp_results = serp_top_10
    st.success(f"✅ Step 2 Complete: {len(serp_top_10)} candidates from SERP scraping")

    main_progress.progress(0.80)

    # ========== STEP 3: MERGE AND FINAL RANKING ==========
    st.write("---")
    st.subheader("🏆 Step 3: Merging and Final Ranking")

    # Combine all candidates
    all_candidates = []
    seen_urls = set()

    # Add semantic results with source tag
    for candidate in semantic_top_10:
        url = candidate.get('linkedin_url', '')
        if url and url not in seen_urls:
            candidate['source'] = 'Semantic Search'
            all_candidates.append(candidate)
            seen_urls.add(url)

    # Add SERP results with source tag
    for candidate in serp_top_10:
        url = candidate.get('linkedinUrl', '') or candidate.get('linkedin_url', '')
        if url and url not in seen_urls:
            candidate['source'] = 'SERP Scraping'
            all_candidates.append(candidate)
            seen_urls.add(url)

    st.info(f"Combined {len(all_candidates)} unique candidates")

    # Re-validate and re-score all candidates asynchronously
    if all_candidates:
        with st.spinner("Re-ranking all candidates (async)..."):
            from validate_v3 import validate_and_score_candidates_async

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            final_candidates, _ = loop.run_until_complete(
                validate_and_score_candidates_async(
                    all_candidates,
                    enhanced_query.get("location", []),
                    enhanced_query.get("job_title", ""),
                    enhanced_query.get("experience", 0),
                    enhanced_query
                )
            )
            loop.close()

        # Final top 10
        final_top_10 = final_candidates[:10]
        st.session_state.final_results = final_top_10
        st.success(f"✅ Final ranking complete: {len(final_top_10)} top candidates")
    else:
        st.warning("No candidates to rank")
        final_top_10 = []

    main_progress.progress(1.0)
    status_text.text("✅ Search complete!")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Semantic Results", len(semantic_top_10))
    with col2:
        st.metric("SERP Results", len(serp_top_10))
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
        # Handle both field name formats
        name = profiles.get('name') or profiles.get('fullName', 'Unknown')

        # Get score for display
        total_score = profiles.get('score', 0)

        with st.expander(f"{idx}. {name} • Score: {total_score}"):
            st.json(profiles)

        with st.expander(
            f"{idx}. {name} • Score: {total_score}", expanded=True
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

                # Determine Open to Work
                experiences = profiles.get('experiences') or profiles.get('experience', [])
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
                    if not linkedin.startswith('http'):
                        linkedin = f"https://{linkedin}"
                    st.markdown(f"**LinkedIn:** [Profile]({linkedin})")

                # Source
                source = profiles.get('source', 'Unknown')
                st.markdown(f"**Source:** {source}")
                
                score_breakdown = profiles.get("score_breakdown")
                st.markdown(f"**Total Exp :** {score_breakdown.get('total_experience_years')}")
                st.markdown(f"**Role Exp :** {score_breakdown.get('role_experience_years')}")

            # --- RIGHT COLUMN ---
            with col2:
                st.markdown(f"### {name}")

                headline = profiles.get('headline')
                if headline:
                    st.markdown(f"*{headline}*")

                # Skills
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

                # About
                about = profiles.get('about')
                if about:
                    st.markdown(
                        "**About:** " + (about[:250] + "..." if len(about) > 250 else about)
                    )

                # Experience
                if experiences:
                    st.markdown("**Experience**")
                    for exp in experiences:
                        if isinstance(exp, dict):
                            title = exp.get("title", "")
                            subtitle = exp.get("subtitle") or exp.get("metadata", "")
                            caption = exp.get("caption", "")
                            st.write(f"• {title} at {subtitle} — {caption}")

                            if exp.get("description"):
                                for desc in exp["description"]:
                                    if isinstance(desc, dict) and "text" in desc:
                                        st.markdown(f"    - {desc['text']}")

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
