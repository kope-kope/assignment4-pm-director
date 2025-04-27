import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent # Using ReAct agent for reasoning
from langchain import hub # To pull standard ReAct prompt

# --- Configuration ---
PRIORITIZED_ROADMAP_FILE = "prioritized_roadmap.txt" # Input from roadmap_planner.py
COMPETITOR_ANALYSIS_OUTPUT_FILE = "competitor_analysis_report.txt" # Output file

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

# --- Main Competitor Analysis Function ---

def run_competitor_analysis(llm_client, search_tool, competitor_name, roadmap_content):
    """
    Runs a LangChain agent to perform competitive analysis.

    Args:
        llm_client: An initialized LangChain LLM client.
        search_tool: An initialized LangChain search tool (e.g., DuckDuckGoSearchRun).
        competitor_name (str): The name of the competitor to analyze (e.g., "X/Twitter").
        roadmap_content (str): The text content of our prioritized roadmap.

    Returns:
        str: The generated competitive analysis report, or an error message.
    """
    print(f"\n--- Performing Competitive Analysis for {competitor_name} ---")
    if not llm_client:
        return "Error: LLM Client not available."
    if not search_tool:
        return "Error: Search tool not available."
    if not roadmap_content:
        return "Error: Roadmap content is missing."

    # Define the tools the agent can use
    tools = [search_tool]

    # Get the ReAct prompt template
    # This prompt is designed for agents that need to reason step-by-step (Thought, Action, Observation)
    prompt = hub.pull("hwchase17/react")

    # Create the ReAct agent
    # This agent uses the LLM to decide which tool to use based on the prompt and reasoning
    agent = create_react_agent(llm_client, tools, prompt)

    # Create the Agent Executor
    # This runs the agent loop: agent thinks -> chooses tool -> runs tool -> gets observation -> repeats
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # Define the input for the agent, including our roadmap and the competitor
    agent_input = f"""
Analyze the competitive landscape for Bluesky social app, focusing on the competitor: {competitor_name}.

Our current prioritized roadmap initiatives are:
--- START ROADMAP ---
{roadmap_content}
--- END ROADMAP ---

Your Task:
1. Use the search tool to find recent (last 3-6 months) feature releases, product updates, strategic shifts, or significant news related to {competitor_name}. Focus on aspects relevant to a social media/microblogging platform.
2. Analyze the search results.
3. Synthesize your findings into a concise competitive analysis report (3-5 paragraphs).
4. Specifically compare the competitor's activities to our roadmap initiatives. Highlight:
    - Any areas where {competitor_name} recently launched features that are on our roadmap (potential threat or validation).
    - Any major initiatives {competitor_name} is pursuing that are *not* on our roadmap (potential blind spots or different strategies).
    - Any areas from our roadmap where {competitor_name} appears to be lagging or inactive.
5. Conclude with key strategic takeaways or potential adjustments Bluesky should consider based on this analysis.
"""

    # Run the agent
    print("Running competitor analysis agent...")
    analysis_report = f"Error: Agent execution failed." # Default
    try:
        # The agent_executor will manage the interaction with the LLM and tools
        result = agent_executor.invoke({"input": agent_input})
        analysis_report = result.get('output', "Agent did not produce standard output.")
        print("Agent execution successful.")
    except Exception as e:
        print(f"Error during agent execution: {e}")
        # The verbose output during execution might give clues
        analysis_report = f"Error during agent execution: {e}"

    return analysis_report

# --- Main execution block ---
if __name__ == "__main__":
    print("Running Competitor Analysis Agent script...")
    load_dotenv() # Load .env file for API key

    # Initialize LLM Client
    llm_competitor = None
    openai_api_key_competitor = os.getenv('OPENAI_API_KEY')
    if openai_api_key_competitor:
        try:
            # GPT-4 might be better for analysis and comparison tasks
            llm_competitor = ChatOpenAI(
                temperature=0.2,
                model_name="gpt-4-turbo-preview",
                openai_api_key=openai_api_key_competitor,
                max_tokens=1000 # Allow more tokens for analysis and report
            )
            print("Competitor Analysis LLM client initialized (using GPT-4 Turbo Preview).")
        except Exception as e:
            print(f"Error initializing Competitor Analysis LLM client: {e}")
            # Fallback if needed
            try:
                 print("Falling back to gpt-3.5-turbo...")
                 llm_competitor = ChatOpenAI(
                     temperature=0.2,
                     model_name="gpt-3.5-turbo",
                     openai_api_key=openai_api_key_competitor,
                     max_tokens=1000
                 )
                 print("Competitor Analysis LLM client initialized (using GPT-3.5 Turbo).")
            except Exception as e2:
                 print(f"Error initializing fallback LLM client: {e2}")
    else:
        print("OpenAI API key not found. Cannot initialize Competitor Analysis LLM client.")

    # Initialize Search Tool
    search = DuckDuckGoSearchRun()
    print("Search tool initialized.")

    # Proceed only if LLM client is ready
    if llm_competitor:
        # Load the prioritized roadmap generated by the previous agent
        roadmap = load_file_content(PRIORITIZED_ROADMAP_FILE)

        # Generate the analysis if roadmap is loaded
        if roadmap:
            # Define the competitor to analyze
            competitor = "X/Twitter" # Can be changed or made dynamic

            analysis_result_text = run_competitor_analysis(
                llm_client=llm_competitor,
                search_tool=search,
                competitor_name=competitor,
                roadmap_content=roadmap
            )

            print("\n" + "="*30 + f" Competitive Analysis Report ({competitor}) " + "="*30)
            print(analysis_result_text)
            print("="*80)

            # Optional: Save analysis report to file
            try:
                with open(COMPETITOR_ANALYSIS_OUTPUT_FILE, "w", encoding="utf-8") as f:
                    f.write(analysis_result_text)
                print(f"\nCompetitor analysis report saved to {COMPETITOR_ANALYSIS_OUTPUT_FILE}")
            except Exception as e:
                print(f"Error saving competitor analysis report to file: {e}")
        else:
            print(f"\nCould not generate competitor analysis because the roadmap file ({PRIORITIZED_ROADMAP_FILE}) was not found or is empty.")
    else:
        print("\nCould not generate competitor analysis because LLM client failed to initialize.")

    print("\nCompetitor Analysis Agent script finished.")

