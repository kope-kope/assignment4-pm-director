import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# --- Configuration ---
NARRATIVE_SUMMARY_FILE = "narrative_summary.txt"
RANKED_TOPICS_FILE = "feedback_topics_summary.csv"
PRIORITIZED_OUTPUT_FILE = "prioritized_roadmap.txt" # Optional output file

# --- Helper Function to Load Files ---

def load_file_content(filepath):
    """Loads text content from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Successfully loaded content from {filepath}")
        return content
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def load_ranked_topics_csv(filepath):
    """Loads ranked topics from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded ranked topics from {filepath}. Shape: {df.shape}")
        if df.empty:
            print("Warning: Ranked topics file is empty.")
            return None
        # Basic validation
        if 'topic' not in df.columns or 'frequency' not in df.columns:
            print("Error: Ranked topics CSV is missing required 'topic' or 'frequency' columns.")
            return None
        return df
    except FileNotFoundError:
        print(f"Error: Ranked topics file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading or validating CSV {filepath}: {e}")
        return None

# --- Main Prioritization Function ---

def prioritize_initiatives(llm_client, narrative_summary, ranked_topics_df, num_topics_to_consider=10):
    """
    Uses an LLM to prioritize initiatives based on narrative summary and ranked topics.

    Args:
        llm_client: An initialized LangChain LLM client.
        narrative_summary (str): The text content of the narrative summary.
        ranked_topics_df (pd.DataFrame): DataFrame with 'topic' and 'frequency'.
        num_topics_to_consider (int): How many top topics to include in the prompt context.

    Returns:
        str: The LLM's response containing the prioritized list, or an error message.
    """
    print("\n--- Prioritizing Initiatives using LLM ---")
    if not llm_client:
        return "Error: LLM Client not available."
    if not narrative_summary:
        return "Error: Narrative summary content is missing."
    if ranked_topics_df is None or ranked_topics_df.empty:
        return "Error: Ranked topics data is missing or empty."

    # Format the top N ranked topics for the prompt
    top_topics = ranked_topics_df.head(num_topics_to_consider)
    ranked_topics_str = "\n".join([f"- {row['topic']} (Frequency: {row['frequency']})" for index, row in top_topics.iterrows()])

    # Define the System Prompt (Instructions for the LLM based on prioritization_prompt)
    system_prompt = """
You are a senior Product Management AI assistant tasked with prioritizing potential development initiatives for the Bluesky social app based on recent user feedback analysis.

Your Task:
Analyze the provided Narrative Summary and the Ranked Topics list. Based on this information, identify and prioritize the Top 3-5 initiatives that should be considered for the development roadmap.

Prioritization Criteria:
Prioritize initiatives based on a combination of:
1. Urgency/Volume (Frequency): Give higher priority to topics mentioned more frequently (see Ranked Topics list).
2. Potential Value/Impact: Assess the likely impact using the Narrative Summary and the nature of the topics. Consider severity (critical bugs?), user experience impact, and relation to core functionality.

Output Format:
List the prioritized initiatives (Top 3-5). For each initiative, provide:
* Initiative: A clear, concise name.
* Priority: (High, Medium, Low).
* Rationale: A brief explanation (1-2 sentences) justifying the priority based on frequency and potential value/impact.

Example Output Structure:
```
Prioritized Initiatives:

1.  **Initiative:** Improve Login Reliability
    **Priority:** High
    **Rationale:** Addresses the most frequent topic identified as a bug in the feedback summary, likely blocking user access and causing significant frustration (high value fix).

2.  **Initiative:** Implement Post Editing Feature
    **Priority:** High
    **Rationale:** Represents a very frequently requested feature, common on other platforms, significantly enhancing core user experience (high value add).

... (up to 5)
```
Focus your analysis ONLY on the provided Narrative Summary and Ranked Topics data.
"""

    # Define the Human Prompt (Input data for the LLM)
    human_prompt_text = f"""
Here is the input data:

1. Narrative Summary:
---
{narrative_summary}
---

2. Ranked Topics (Top {num_topics_to_consider} by Frequency):
---
{ranked_topics_str}
---

Please provide the prioritized list of initiatives based on these inputs and the criteria outlined in your instructions.
"""

    # Call the LLM
    print("Calling LLM for prioritization...")
    prioritized_list = "Error: LLM call failed." # Default
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt_text)
        ]
        # Use a model potentially better suited for reasoning/longer output if needed
        # Ensure max_tokens in LLM init is sufficient
        response = llm_client.invoke(messages)
        prioritized_list = response.content
        print("LLM call successful.")
    except Exception as e:
        print(f"Error during LLM call for prioritization: {e}")
        prioritized_list = f"Error during LLM call: {e}"

    return prioritized_list

# --- Main execution block ---
if __name__ == "__main__":
    print("Running Roadmap Planner script...")
    load_dotenv() # Load .env file for API key

    # Initialize LLM Client
    llm_planner = None
    openai_api_key_planner = os.getenv('OPENAI_API_KEY')
    if openai_api_key_planner:
        try:
            llm_planner = ChatOpenAI(
                temperature=0.3, # Slightly higher temp for reasoning but still focused
                model_name="gpt-4-turbo-preview", # Consider GPT-4 for better reasoning
                openai_api_key=openai_api_key_planner,
                max_tokens=500 # Allow reasonable length for the prioritized list
            )
            print("Planner LLM client initialized (using GPT-4 Turbo Preview).")
        except Exception as e:
            print(f"Error initializing Planner LLM client: {e}")
            # Fallback to gpt-3.5 if gpt-4 fails or key doesn't support it
            try:
                 print("Falling back to gpt-3.5-turbo...")
                 llm_planner = ChatOpenAI(
                     temperature=0.3,
                     model_name="gpt-3.5-turbo",
                     openai_api_key=openai_api_key_planner,
                     max_tokens=500
                 )
                 print("Planner LLM client initialized (using GPT-3.5 Turbo).")
            except Exception as e2:
                 print(f"Error initializing fallback LLM client: {e2}")

    else:
        print("OpenAI API key not found. Cannot initialize Planner LLM client.")

    # Proceed only if LLM client is ready
    if llm_planner:
        # Load the necessary input files
        narrative = load_file_content(NARRATIVE_SUMMARY_FILE)
        ranked_topics = load_ranked_topics_csv(RANKED_TOPICS_FILE)

        # Generate the prioritized list if inputs are valid
        if narrative and ranked_topics is not None:
            prioritized_initiatives_text = prioritize_initiatives(
                llm_client=llm_planner,
                narrative_summary=narrative,
                ranked_topics_df=ranked_topics
            )

            print("\n" + "="*30 + " Prioritized Roadmap Initiatives " + "="*30)
            print(prioritized_initiatives_text)
            print("="*80)

            # Optional: Save prioritized list to file
            try:
                with open(PRIORITIZED_OUTPUT_FILE, "w", encoding="utf-8") as f:
                    f.write(prioritized_initiatives_text)
                print(f"\nPrioritized initiatives saved to {PRIORITIZED_OUTPUT_FILE}")
            except Exception as e:
                print(f"Error saving prioritized initiatives to file: {e}")
        else:
            print("\nCould not generate prioritized list due to missing input files.")
    else:
        print("\nCould not generate prioritized list because LLM client failed to initialize.")

    print("\nRoadmap Planner script finished.")

