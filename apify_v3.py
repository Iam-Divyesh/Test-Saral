
from apify_client import ApifyClient
from dotenv import load_dotenv
import os

load_dotenv()

APIFY_API_KEY = os.getenv("APIFY_API_TOKEN")
client = ApifyClient(APIFY_API_KEY)

def apify_call(linkedin_urls: dict):
    """Scrape LinkedIn profiles using Apify"""
    if not linkedin_urls:
        return []

    list_links = list(linkedin_urls.values())
    
    run_input = {
        "profileUrls": list_links
    }

    try:
        run = client.actor("2SyF0bVxmgGr8IVCZ").call(run_input=run_input)
        
        cleaned_profiles = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            cleaned_profiles.append(item)
        
        return cleaned_profiles
    except Exception as e:
        print(f"Apify error: {e}")
        return []
