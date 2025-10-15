import psycopg2
from psycopg2 import pool
import json
from datetime import datetime

# Create connection pool for reuse
connection_pool = None

def get_connection_pool():
    """Get or create connection pool"""
    global connection_pool
    if connection_pool is None:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            host="43.205.29.110",
            port=5432,
            database="saral_ai",
            user="saral_user",
            password="8k$ScgT97y9£>D"
        )
    return connection_pool

def get_connection():
    """Get connection from pool"""
    pool = get_connection_pool()
    return pool.getconn()

def return_connection(conn):
    """Return connection to pool"""
    pool = get_connection_pool()
    pool.putconn(conn)

def get_data_with_embeddings():
    """Fetch all candidates with embeddings from database"""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Use server-side cursor for large datasets
        cur.execute("""
            SELECT id, name, location, email, linkedin_url, headline, 
                   skills, about, experience, profile_pic, embedding
            FROM temp_profiles
            WHERE embedding IS NOT NULL
        """)

        # Fetch all at once (faster for smaller datasets)
        rows = cur.fetchall()

        # Use list comprehension for faster iteration
        candidates = [
            {
                'id': row[0],
                'name': row[1],
                'location': row[2],
                'email': row[3],
                'linkedin_url': row[4],
                'headline': row[5],
                'skills': row[6],
                'about': row[7],
                'experience': row[8],
                'profile_pic': row[9],
                'embedding': row[10]
            }
            for row in rows
        ]

        cur.close()
        return_connection(conn)
        return candidates
    except Exception as e:
        print(f"Database error: {e}")
        if conn:
            return_connection(conn)
        return []

# Placeholder for get_query_embedding function, assuming it exists elsewhere
def get_query_embedding(text):
    # In a real scenario, this would call an embedding model
    # For demonstration, returning a dummy embedding
    if text:
        return [0.1] * 1536  # Example dimension
    return None

def store_apify_profiles(profiles):
    """Store Apify scraped profiles in database with batch insert"""
    if not profiles:
        return

    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Prepare batch data
        batch_data = []
        for profile in profiles:
            linkedin_url = profile.get('linkedinUrl')
            if not linkedin_url:
                continue

            # Generate embedding for this profile
            embedding_text = f"{profile.get('fullName', '')} {profile.get('headline', '')} {profile.get('about', '')}"
            embedding = get_query_embedding(embedding_text)

            batch_data.append((
                profile.get('fullName'),
                profile.get('addressWithCountry'),
                profile.get('email'),
                linkedin_url,
                profile.get('headline'),
                json.dumps(profile.get('skills', [])),
                profile.get('about'),
                json.dumps(profile.get('experiences', [])),
                profile.get('profilePic'),
                json.dumps(embedding) if embedding else None
            ))

        if batch_data:
            # Use execute_values for batch insert (much faster)
            from psycopg2.extras import execute_values
            execute_values(cur, """
                INSERT INTO temp_profiles (
                    name, location, email, linkedin_url, headline, 
                    skills, about, experience, profile_pic, embedding
                ) VALUES %s
                ON CONFLICT (linkedin_url) DO UPDATE SET
                    name = EXCLUDED.name,
                    location = EXCLUDED.location,
                    email = EXCLUDED.email,
                    headline = EXCLUDED.headline,
                    skills = EXCLUDED.skills,
                    about = EXCLUDED.about,
                    experience = EXCLUDED.experience,
                    profile_pic = EXCLUDED.profile_pic,
                    embedding = EXCLUDED.embedding
            """, batch_data)

        conn.commit()
        cur.close()
        return_connection(conn)
    except Exception as e:
        print(f"Store error: {e}")
        if conn:
            return_connection(conn)
