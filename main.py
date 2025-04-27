import os
import re
import pandas as pd
import json
import time
from dotenv import load_dotenv
from atproto import Client as BskyClient
from atproto import models as BskyModels # Use alias for clarity
from openai import OpenAI
# import pprint # No longer needed after removing pprint.pprint call
from google_play_scraper import reviews_all, Sort
from langdetect import detect, LangDetectException
# Imports for potential future Langchain use (kept commented if not used now)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
import logging # Using logging instead of print for better control

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
load_dotenv()
logging.info("Loading environment variables...")

# --- Configuration ---
bluesky_handle = os.getenv('BLUESKY_HANDLE')
bluesky_password = os.getenv('BLUESKY_APP_PASSWORD')
openai_api_key = os.getenv('OPENAI_API_KEY')
appstore_app_id = os.getenv('APPSTORE_APP_ID')
playstore_app_id = os.getenv('PLAYSTORE_APP_ID')

# --- Initialize API Clients ---
bsky_client = None
if bluesky_handle and bluesky_password:
    try:
        bsky_client = BskyClient()
        bsky_client.login(bluesky_handle, bluesky_password)
        logging.info("Successfully logged into Bluesky.")
    except Exception as e:
        logging.error(f"Error logging into Bluesky: {e}")
else:
    logging.warning("Bluesky handle or password not found in environment variables. Bluesky collection skipped.")

llm = None
if openai_api_key:
  try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
         temperature=0,
         openai_api_key=openai_api_key,
         max_tokens=100
         )
  except Exception as e:
    logging.error(f"Error initializing LangChain OpenAI client: {e}")
else:
    logging.warning("OpenAI API key not found. OpenAI dependent features disabled.")


# --- Collection Functions ---

def collect_bluesky_feedback(client):
    """Collects feedback posts from Bluesky matching specific hashtags."""
    logging.info("Collecting Bluesky feedback...")
    all_posts_data = []
    if not client:
        logging.warning("Bluesky client not available. Skipping collection.")
        return pd.DataFrame(all_posts_data)

    search_query = "#bugreport #featurerequest"
    search_limit = 100 # Consider making this configurable
    logging.info(f"Searching Bluesky posts for query: '{search_query}' (limit: {search_limit})")

    try:
        # Using the correct API call structure with Params object
        search_params = BskyModels.AppBskyFeedSearchPosts.Params(
            q=search_query,
            limit=search_limit
        )
        response = client.app.bsky.feed.search_posts(params=search_params)

        if response and hasattr(response, 'posts'):
            logging.info(f"Found {len(response.posts)} potential posts from Bluesky.")
            for post in response.posts:
                try:
                    post_text = getattr(post.record, 'text', '')
                    author_handle = getattr(post.author, 'handle', 'N/A')
                    created_at = getattr(post.record, 'created_at', None)
                    like_count = getattr(post, 'like_count', 0)
                    uri = getattr(post, 'uri', None)
                    if not uri: continue

                    post_url = f"https://bsky.app/profile/{author_handle}/post/{uri.split('/')[-1]}"

                    all_posts_data.append({
                        'source': 'Bluesky',
                        'id': uri,
                        'text': post_text,
                        'author': author_handle,
                        'timestamp': created_at,
                        'likes': like_count,
                        'url': post_url,
                        'rating': None # Consistent column schema
                    })
                except Exception as e:
                    logging.warning(f"Could not parse a Bluesky post fully: {e} - URI: {getattr(post, 'uri', 'N/A')}")
        else:
            logging.info("No posts found or unexpected response format from Bluesky.")

    except Exception as e:
        logging.error(f"Error during Bluesky search: {e}")

    logging.info(f"Collected {len(all_posts_data)} posts from Bluesky.")
    return pd.DataFrame(all_posts_data)

def collect_app_store_reviews(appstore_id, playstore_id):
    """Collects reviews from Apple App Store and Google Play Store."""
    logging.info("Collecting App Store & Play Store reviews...")
    all_reviews_data = []
    max_reviews_per_store = 1000 # Limit reviews for faster processing

    # --- Apple App Store (Commented out as per original code) ---
    # if appstore_id:
    #     try:
    #         logging.info(f"Fetching Apple App Store reviews for ID: {appstore_id}...")
    #         app_store = AppStore(country='gb', app_name='bluesky-social', app_id=appstore_id) # Consider making country/app_name params or env vars
    #         app_store.review(how_many=max_reviews_per_store)
    #         apple_reviews = app_store.reviews
    #         for review in apple_reviews:
    #              all_reviews_data.append({
    #                 'source': 'AppStore',
    #                 'id': review.get('id', None),
    #                 'text': review.get('review', ''),
    #                 'author': review.get('userName', 'N/A'),
    #                 'timestamp': review.get('date', None),
    #                 'rating': review.get('rating', None),
    #                 'url': None,
    #                 'likes': None
    #             })
    #         logging.info(f"Collected {len(apple_reviews)} reviews from Apple App Store.")
    #     except Exception as e:
    #         logging.error(f"Error fetching Apple App Store reviews: {e}")
    # else:
    #     logging.warning("App Store ID not provided. Skipping Apple App Store.")
    # --- End Apple App Store ---

    # --- Google Play Store ---
    if playstore_id:
        try:
            logging.info(f"Fetching Google Play Store reviews for ID: {playstore_id}...")
            # reviews_all can be slow. Consider using `reviews` with `count` if performance is an issue.
            play_reviews = reviews_all(
                playstore_id,
                sleep_milliseconds=500, # Be polite to the API
                lang='en',             # Consider making lang/country params or env vars
                country='us',
                sort=Sort.NEWEST,
            )
            # Apply limit after fetching if using reviews_all
            if len(play_reviews) > max_reviews_per_store:
                 logging.info(f"Limiting Play Store reviews from {len(play_reviews)} to {max_reviews_per_store}.")
                 play_reviews = play_reviews[:max_reviews_per_store]


            for review in play_reviews:
                all_reviews_data.append({
                    'source': 'PlayStore',
                    'id': review.get('reviewId', None),
                    'text': review.get('content', ''),
                    'author': review.get('userName', 'N/A'),
                    'timestamp': review.get('at', None),
                    'rating': review.get('score', None),
                    'url': review.get('url', None),
                    'likes': review.get('thumbsUpCount', None)
                })
            logging.info(f"Collected {len(play_reviews)} reviews from Google Play Store.")
        except Exception as e:
            logging.error(f"Error fetching Google Play Store reviews: {e}")
    else:
        logging.warning("Play Store ID not provided. Skipping Google Play Store.")
    # --- End Google Play Store ---

    logging.info(f"Collected total {len(all_reviews_data)} reviews from stores.")
    return pd.DataFrame(all_reviews_data)

# --- Preprocessing Functions ---

def detect_lang_safe(text):
    """Safely detects language, returning 'unknown' on failure."""
    if not isinstance(text, str) or not text.strip():
        return 'unknown' # Handle non-string or empty input
    try:
        return detect(text)
    except LangDetectException:
        # logging.debug(f"Language detection failed for text snippet: {text[:50]}...") # Optional debug log
        return 'unknown'

def preprocess_text(text):
    """Cleans text by lowercasing, removing URLs, and normalizing whitespace."""
    if not isinstance(text, str):
        return "" # Handle non-string input
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(df):
    """Preprocesses the combined feedback DataFrame."""
    logging.info("Preprocessing combined feedback data...")
    if df.empty:
        logging.warning("Input DataFrame is empty. Skipping preprocessing.")
        return df

    initial_rows = len(df)
    logging.info(f"Initial rows: {initial_rows}")

    # 1. Handle Missing/Empty Text
    df.dropna(subset=['text'], inplace=True)
    df = df[df['text'].str.strip() != '']
    rows_after_dropna = len(df)
    if rows_after_dropna < initial_rows:
        logging.info(f"Rows after dropping empty text: {rows_after_dropna} ({initial_rows - rows_after_dropna} removed)")
    if df.empty:
        logging.warning("No data left after removing entries with empty text.")
        return df

    # 2. Clean Text
    df['text_cleaned'] = df['text'].apply(preprocess_text)

    # 3. Filter Language (keep only English)
    df['language'] = df['text_cleaned'].apply(detect_lang_safe)
    df_processed = df[df['language'] == 'en'].copy()
    rows_after_lang_filter = len(df_processed)
    if rows_after_lang_filter < rows_after_dropna:
         logging.info(f"Rows after filtering for English: {rows_after_lang_filter} ({rows_after_dropna - rows_after_lang_filter} non-English removed)")

    if df_processed.empty:
        logging.warning("No English feedback found after filtering.")
        return df_processed

    # Select and potentially reorder columns for clarity
    # Keep original text for reference if needed
    final_columns = ['source', 'id', 'text', 'text_cleaned', 'author', 'timestamp', 'likes', 'rating', 'url', 'language']
    df_processed = df_processed[final_columns].copy()

    logging.info(f"Preprocessing complete. Final rows: {len(df_processed)}")
    return df_processed

# --- Analysis Function (Placeholder - Ensure it returns BOTH DataFrames) ---
def analyze_and_rank(df_processed, client):
    """
    Placeholder for analyzing feedback, assigning topics, and aggregating.

    IMPORTANT: This function MUST return two DataFrames:
    1. aggregated_topics_df: DataFrame with columns like ['topic', 'frequency']
    2. processed_with_topics_df: The input df_processed DataFrame with a new
                                   'topic' column added based on the analysis.
    """
    logging.info("Analyzing feedback and assigning topics...")
    if client:
         logging.info("OpenAI client is available for analysis.")
    else:
         logging.warning("OpenAI client not available for analysis. Using basic keyword matching.")

    # --- Mock Analysis ---
    # This is a simplified example. Replace with your actual LLM logic.
    # 1. Add 'topic' column to the processed DataFrame
    # Initialize with a default topic or None
    df_processed['topic'] = 'uncategorized'

    # Example: Assign topics based on keywords (replace/enhance with LLM)
    df_processed.loc[df_processed['text_cleaned'].str.contains('bug|crash|error|fix|issue|fail|problem', case=False, na=False), 'topic'] = 'potential_bug'
    df_processed.loc[df_processed['text_cleaned'].str.contains('feature|request|suggest|idea|add|implement|improve|enhancement', case=False, na=False), 'topic'] = 'feature_request'
    df_processed.loc[df_processed['text_cleaned'].str.contains('login|sign in|password|authenticate', case=False, na=False), 'topic'] = 'login_issue'
    df_processed.loc[df_processed['text_cleaned'].str.contains('ui|ux|interface|design|layout|look', case=False, na=False), 'topic'] = 'ui_ux'
    # ... add more rules or use LLM classification ...

    # Keep only rows where a specific topic was assigned (optional, depends on needs)
    # df_processed_with_topics = df_processed[df_processed['topic'] != 'uncategorized'].copy()
    df_processed_with_topics = df_processed.copy() # Keep all for now

    # 2. Aggregate topics to get frequency counts
    if 'topic' in df_processed_with_topics.columns and not df_processed_with_topics['topic'].isnull().all():
        aggregated_topics_df = df_processed_with_topics['topic'].value_counts().reset_index()
        aggregated_topics_df.columns = ['topic', 'frequency']
        # Sort by frequency
        aggregated_topics_df = aggregated_topics_df.sort_values('frequency', ascending=False).reset_index(drop=True)
        logging.info(f"Aggregated {len(aggregated_topics_df)} topics.")
    else:
         logging.warning("No topics were assigned or found. Creating empty aggregated topics DataFrame.")
         aggregated_topics_df = pd.DataFrame(columns=['topic', 'frequency'])


    # --- End Mock Analysis ---

    # Return BOTH DataFrames
    return aggregated_topics_df, df_processed_with_topics


# --- Utility Function to Save CSV ---
def save_results_to_csv(df, filename="output.csv"):
    """Saves the DataFrame to a CSV file."""
    if df is None or not isinstance(df, pd.DataFrame):
         logging.warning(f"Invalid data provided for saving to {filename}. Skipping.")
         return
    if df.empty:
        logging.warning(f"DataFrame is empty. Skipping save to {filename}.")
        return

    try:
        df.to_csv(filename, index=False, encoding='utf-8')
        logging.info(f"Successfully saved data to {filename}")
    except Exception as e:
        logging.error(f"Failed to save data to {filename}: {e}")


# --- Main Execution Logic ---
def main():
    logging.info("Starting User Feedback Agent...")

    # --- Check Prerequisites ---
    # (Simplified check: ensure at least one source ID is present)
    # Add checks for API keys if specific steps depend on them.
    if not playstore_app_id and not appstore_app_id and not (bluesky_handle and bluesky_password):
         logging.error("No data sources configured (App/Play Store ID missing and/or Bluesky credentials missing). Exiting.")
         return

    # --- Step 1: Collection ---
    df_bsky = collect_bluesky_feedback(bsky_client)
    df_reviews = collect_app_store_reviews(appstore_app_id, playstore_app_id)

    # Combine data
    if df_bsky.empty and df_reviews.empty:
        logging.warning("No feedback collected from any source.")
        return
    elif df_bsky.empty:
        df_combined = df_reviews
    elif df_reviews.empty:
        df_combined = df_bsky
    else:
        df_combined = pd.concat([df_bsky, df_reviews], ignore_index=True)
    logging.info(f"Total raw feedback items collected: {len(df_combined)}")

    # --- Step 2: Preprocessing ---
    df_processed = preprocess_data(df_combined)
    if df_processed.empty:
        logging.warning("No processable feedback data left after preprocessing.")
        return
    # Save intermediate processed data (optional)
    # save_results_to_csv(df_processed, "temp_processed_feedback.csv")


    # --- Step 3: Analysis & Topic Assignment ---
    try:
        # Expect analyze_and_rank to return:
        # 1. Aggregated topics (topic, frequency)
        # 2. Processed data with 'topic' column added
        df_aggregated_topics, df_processed_with_topics = analyze_and_rank(df_processed, llm)
    except Exception as e: # Broader catch during analysis development
         logging.error(f"An error occurred during the analysis step: {e}", exc_info=True) # Log traceback
         logging.info("Saving processed data before exiting due to analysis error.")
         save_results_to_csv(df_processed, "processed_feedback_before_error.csv")
         return # Exit if analysis fails catastrophically

    # Validate analysis results are DataFrames
    if not isinstance(df_aggregated_topics, pd.DataFrame) or not isinstance(df_processed_with_topics, pd.DataFrame):
         logging.error("Analysis function did not return two valid pandas DataFrames. Saving pre-analysis data.")
         save_results_to_csv(df_processed, "processed_feedback_before_error.csv")
         return

    # --- Step 4: Output Generation (Saving CSVs) ---
    logging.info("Saving analysis results...")

    # Save the AGGREGATED topics summary
    save_results_to_csv(df_aggregated_topics, filename="feedback_topics_summary.csv")

    # Save the PROCESSED feedback including the assigned TOPIC for each item
    save_results_to_csv(df_processed_with_topics, filename="processed_feedback_with_topics.csv")

    # --- Report Generation Removed ---
    # final_report_text = generate_report(df_aggregated_topics, df_processed_with_topics, openai_client)
    # ... (code to save final_report_text removed) ...

    # --- Step 5: Memory (Placeholder) ---
    # logging.info("Placeholder: Saving to database/memory...")

    logging.info("Agent finished successfully. Output files: feedback_topics_summary.csv, processed_feedback_with_topics.csv")


if __name__ == "__main__":
    main()

# Remove the generate_report function entirely if it's no longer needed
# def generate_report(...):
#    ...