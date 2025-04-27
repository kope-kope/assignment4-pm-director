import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI # Need this for client init

# --- Configuration ---
ANALYZED_DATA_FILE = "processed_feedback_with_topics.csv"
RANKED_TOPICS_FILE = "feedback_topics_summary.csv"
NARRATIVE_OUTPUT_FILE = "narrative_summary.txt" # Optional output file

def generate_narrative_summary(llm_client, analyzed_csv_filepath, ranked_topics_csv_filepath):
    """
    Loads analyzed feedback data and ranked topics from CSV files and uses an LLM
    to generate a narrative summary synthesizing both.

    Args:
        llm_client: An initialized LangChain LLM client (e.g., ChatOpenAI).
        analyzed_csv_filepath (str): Path to CSV with detailed analyzed feedback.
        ranked_topics_csv_filepath (str): Path to CSV with ranked topics and frequencies.

    Returns:
        str: The generated narrative summary string, or an error message.
    """
    print(f"\n--- Generating Narrative Summary using {analyzed_csv_filepath} and {ranked_topics_csv_filepath} ---")
    if not llm_client:
        print("LLM client not initialized. Cannot generate summary.")
        return "Error: LLM client not available."

    # --- 1. Load Data ---
    df_analyzed = None
    df_ranked_topics = None

    try:
        df_analyzed = pd.read_csv(analyzed_csv_filepath)
        print(f"Successfully loaded analyzed data from {analyzed_csv_filepath}. Shape: {df_analyzed.shape}")
        if df_analyzed.empty:
            print("Warning: Loaded analyzed data file is empty.")
        elif 'category' not in df_analyzed.columns or 'topic' not in df_analyzed.columns or 'text' not in df_analyzed.columns:
             print(f"Warning: Loaded CSV {analyzed_csv_filepath} is missing required columns ('category', 'topic', 'text'). Examples may be unavailable.")
             # Treat as if empty for example fetching
             df_analyzed = pd.DataFrame()
    except FileNotFoundError:
        print(f"Warning: Analyzed data file not found at {analyzed_csv_filepath}. Proceeding without examples/detailed counts.")
        df_analyzed = pd.DataFrame() # Use empty DataFrame
    except Exception as e:
        print(f"Error loading or validating CSV {analyzed_csv_filepath}: {e}")
        return f"Critical Error loading analyzed CSV: {e}"

    try:
        df_ranked_topics = pd.read_csv(ranked_topics_csv_filepath)
        print(f"Successfully loaded ranked topics from {ranked_topics_csv_filepath}. Shape: {df_ranked_topics.shape}")
        if df_ranked_topics.empty:
            print("Error: Loaded ranked topics data is empty. Cannot determine key themes.")
            return "Error: No ranked topics found to summarize."
        if 'topic' not in df_ranked_topics.columns or 'frequency' not in df_ranked_topics.columns:
             raise ValueError("Loaded ranked topics CSV is missing required 'topic' or 'frequency' columns.")
    except FileNotFoundError:
        print(f"Error: Ranked topics file not found at {ranked_topics_csv_filepath}.")
        return f"Error: File {ranked_topics_csv_filepath} not found. Please ensure analysis stage completed successfully."
    except Exception as e:
        print(f"Error loading or validating CSV {ranked_topics_csv_filepath}: {e}")
        return f"Critical Error loading ranked topics CSV: {e}"

    # --- 2. Synthesize Insights for Prompt ---
    total_feedback_items = len(df_analyzed) if not df_analyzed.empty else "Unknown (analyzed data unavailable)"
    num_bugs = "Unknown"
    num_features = "Unknown"
    examples = []
    example_section = "No specific examples available from analyzed data."

    if not df_analyzed.empty:
        # Ensure required columns exist before using them
        if all(col in df_analyzed.columns for col in ['category', 'topic', 'text']):
             df_analyzed.dropna(subset=['category', 'topic', 'text'], inplace=True) # Clean before selecting
             num_bugs = len(df_analyzed[df_analyzed['category'] == 'Bug Report'])
             num_features = len(df_analyzed[df_analyzed['category'] == 'Feature Request'])

             bug_reports = df_analyzed[df_analyzed['category'] == 'Bug Report']
             feature_requests = df_analyzed[df_analyzed['category'] == 'Feature Request']

             if not bug_reports.empty:
                 examples.append(f"Example Bug Report: \"{bug_reports['text'].iloc[0][:150]}...\"")
             if not feature_requests.empty:
                 examples.append(f"Example Feature Request: \"{feature_requests['text'].iloc[0][:150]}...\"")
             example_section = "\n".join(examples) if examples else "No specific bug/feature examples found in analyzed data."
        else:
            print("Warning: Could not calculate exact bug/feature counts or find examples due to missing columns in analyzed data.")


    # Get top topics directly from the ranked topics file
    top_overall_topics_list = df_ranked_topics['topic'].head(5).tolist()
    top_overall_topics = ", ".join(top_overall_topics_list) if top_overall_topics_list else "None identified."

    # --- 3. Craft the Storytelling Prompt ---
    story_prompt_text = f"""
You are a Product Management assistant summarizing user feedback for the Bluesky social app based on a recent analysis.

Analysis Summary:
- Total feedback items analyzed (approximate): {total_feedback_items}
- Number of Bug Reports identified (approximate): {num_bugs}
- Number of Feature Requests identified (approximate): {num_features}
- Top 5 Most Frequent Topics Overall (Bugs & Features): {top_overall_topics}

Selected Feedback Examples (if available):
{example_section}

Task:
Based *only* on the analysis summary and examples provided above, write a concise (roughly one page or 3-4 paragraphs) narrative summary "telling the story" of this user feedback. Highlight the main themes derived from the top topics, mentioning common pain points (bugs) and desires (feature requests) suggested by these topics. Maintain a neutral and informative tone. Start with an overview and then elaborate on the key themes.
"""

    # --- 4. Call the LLM ---
    print("Generating narrative summary via LLM...")
    narrative_summary = "Error generating summary." # Default
    try:
        # Ensure the LLM client used here has sufficient max_tokens for a story
        response = llm_client.invoke([HumanMessage(content=story_prompt_text)])
        narrative_summary = response.content
        print("Narrative summary generated successfully.")
    except Exception as e:
        print(f"  Error during LLM call for narrative summary: {e}")
        narrative_summary = f"Error generating summary: {e}"

    # --- 5. Return the Story ---
    return narrative_summary

# --- Main execution block to allow running this file directly ---
if __name__ == "__main__":
    print("Running Narrative Generator script...")
    load_dotenv() # Load .env file for API key

    # Initialize LLM Client
    llm_narrative = None
    openai_api_key_narrative = os.getenv('OPENAI_API_KEY')
    if openai_api_key_narrative:
        try:
            llm_narrative = ChatOpenAI(
                temperature=0.7, # Slightly higher temp for more narrative style
                model_name="gpt-3.5-turbo", # Or a model known for better long-form generation
                openai_api_key=openai_api_key_narrative,
                max_tokens=700 # Increased tokens for potentially longer story
            )
            print("Narrative LLM client initialized.")
        except Exception as e:
            print(f"Error initializing Narrative LLM client: {e}")
    else:
        print("OpenAI API key not found. Cannot generate narrative.")

    # Proceed only if LLM client is ready
    if llm_narrative:
        # Generate the summary using the function above
        summary = generate_narrative_summary(
            llm_client=llm_narrative,
            analyzed_csv_filepath=ANALYZED_DATA_FILE,
            ranked_topics_csv_filepath=RANKED_TOPICS_FILE
        )

        print("\n" + "="*30 + " Generated Narrative Summary " + "="*30)
        print(summary)
        print("="*80)

        # Optional: Save narrative summary to file
        try:
            with open(NARRATIVE_OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"\nNarrative summary saved to {NARRATIVE_OUTPUT_FILE}")
        except Exception as e:
            print(f"Error saving narrative summary to file: {e}")
    else:
        print("\nCould not generate narrative summary because LLM client failed to initialize.")

    print("\nNarrative Generator script finished.")
