import os
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Optional
import operator
import json # For pretty printing state
import argparse # Import argparse for command-line arguments

# --- LangChain/LangGraph Imports ---
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage # For prompts

# --- Import functions from our refactored modules ---
try:
    from narrative_generator import generate_narrative_summary
    from roadmap_planner import prioritize_initiatives, load_file_content, load_ranked_topics_csv
    from competitor_agent import run_competitor_analysis
    from utils import generate_pdf_document, send_email_with_attachments # Import new utils
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure narrative_generator.py, roadmap_planner.py, competitor_agent.py, and utils.py exist and are importable.")
    # Define dummy functions if imports fail
    def generate_narrative_summary(llm, **kwargs): return "Dummy Narrative Summary"
    def prioritize_initiatives(llm, **kwargs): return "Dummy Prioritized Roadmap"
    def run_competitor_analysis(llm, search, **kwargs): return "Dummy Competitor Analysis"
    def load_file_content(filepath): return f"Dummy content for {filepath}" if os.path.exists(filepath) else None
    def load_ranked_topics_csv(filepath): return pd.DataFrame({'topic':['dummy'], 'frequency':[1]}) if os.path.exists(filepath) else None
    def generate_pdf_document(content, filename, title): return filename if content else f"Error: No content for {filename}"
    def send_email_with_attachments(**kwargs): return "Dummy Email Sent Successfully"


# --- Configuration ---
# Input files
ANALYZED_DATA_FILE = "processed_feedback_with_topics.csv"
RANKED_TOPICS_FILE = "feedback_topics_summary.csv"
# Output files (PDFs)
NARRATIVE_OUTPUT_PDF = "narrative_summary.pdf"
PRIORITIZED_OUTPUT_PDF = "prioritized_roadmap.pdf"
COMPETITOR_ANALYSIS_OUTPUT_PDF = "competitor_analysis_report.pdf"

# --- Define the State for the Graph ---

class ReportGenerationState(TypedDict):
    """
    Represents the state passed between nodes in the reporting graph.
    Includes paths for generated PDFs and email details.
    """
    # Inputs & Clients
    llm_client: ChatOpenAI
    search_tool: DuckDuckGoSearchRun
    analyzed_data_path: str
    ranked_topics_path: str
    recipient_email: str # Recipient email now comes from input args

    # Intermediate/Generated Data
    narrative_summary: Optional[str]
    ranked_topics_df: Optional[pd.DataFrame]
    prioritized_roadmap: Optional[str]
    competitor_analysis: Optional[str]
    email_subject: Optional[str]
    email_body: Optional[str]

    # PDF Paths (generated)
    narrative_pdf_path: Optional[str]
    roadmap_pdf_path: Optional[str]
    competitor_pdf_path: Optional[str]

    # Final Status/Error
    final_message: Optional[str]
    error_message: Optional[str]

# --- Define Graph Nodes ---
# (Node functions: load_initial_data_node, run_narrative_node, run_planner_node,
#  run_competitor_node, generate_email_body_node, send_email_node remain the same
#  as in the previous version - they already use the recipient_email from the state)

def load_initial_data_node(state: ReportGenerationState) -> dict:
    """Loads the ranked topics CSV needed early."""
    print("--- Node: Loading Initial Data ---")
    ranked_topics_df = load_ranked_topics_csv(state["ranked_topics_path"])
    if ranked_topics_df is None:
        return {"error_message": f"Failed to load ranked topics from {state['ranked_topics_path']}"}
    return {"ranked_topics_df": ranked_topics_df, "error_message": None} # Clear previous errors

def run_narrative_node(state: ReportGenerationState) -> dict:
    """Generates narrative summary and saves as PDF."""
    print("--- Node: Generating Narrative Summary ---")
    if state.get("error_message"): return {} # Skip if previous error
    summary = generate_narrative_summary(
        llm_client=state["llm_client"],
        analyzed_csv_filepath=state["analyzed_data_path"],
        ranked_topics_csv_filepath=state["ranked_topics_path"]
    )
    if "Error:" in summary:
        return {"error_message": f"Narrative generation failed: {summary}"}

    # Generate PDF instead of TXT
    pdf_path = generate_pdf_document(
        content=summary,
        filename=NARRATIVE_OUTPUT_PDF,
        title="User Feedback Narrative Summary"
    )
    if "Error:" in pdf_path:
         return {"error_message": f"Narrative PDF generation failed: {pdf_path}"}

    return {"narrative_summary": summary, "narrative_pdf_path": pdf_path, "error_message": None}

def run_planner_node(state: ReportGenerationState) -> dict:
    """Generates prioritized roadmap and saves as PDF."""
    print("--- Node: Prioritizing Roadmap ---")
    if state.get("error_message"): return {} # Skip if previous error
    if not state.get("narrative_summary") or state.get("ranked_topics_df") is None:
        return {"error_message": "Missing narrative summary or ranked topics for planner."}

    roadmap = prioritize_initiatives(
        llm_client=state["llm_client"],
        narrative_summary=state["narrative_summary"],
        ranked_topics_df=state["ranked_topics_df"]
    )
    if "Error:" in roadmap:
        return {"error_message": f"Roadmap prioritization failed: {roadmap}"}

    # Generate PDF
    pdf_path = generate_pdf_document(
        content=roadmap,
        filename=PRIORITIZED_OUTPUT_PDF,
        title="Prioritized Roadmap Initiatives"
    )
    if "Error:" in pdf_path:
         return {"error_message": f"Roadmap PDF generation failed: {pdf_path}"}

    return {"prioritized_roadmap": roadmap, "roadmap_pdf_path": pdf_path, "error_message": None}

def run_competitor_node(state: ReportGenerationState) -> dict:
    """Runs competitor analysis and saves as PDF."""
    print("--- Node: Analyzing Competitor ---")
    if state.get("error_message"): return {} # Skip if previous error

    # Use roadmap content from state if available, otherwise try loading file
    roadmap_content = state.get("prioritized_roadmap")
    if not roadmap_content:
        # Try loading the text content used to generate the PDF if needed
        roadmap_content = load_file_content(PRIORITIZED_OUTPUT_FILE.replace(".pdf", ".txt")) # Assuming source text might be saved
        if not roadmap_content:
             # Last resort: try loading the PDF source file itself if it exists
             roadmap_content = load_file_content(PRIORITIZED_OUTPUT_FILE)
             if not roadmap_content:
                 return {"error_message": "Missing prioritized roadmap for competitor analysis."}

    analysis = run_competitor_analysis(
        llm_client=state["llm_client"],
        search_tool=state["search_tool"],
        competitor_name="X/Twitter", # Or get dynamically
        roadmap_content=roadmap_content
    )
    if "Error:" in analysis:
        return {"error_message": f"Competitor analysis failed: {analysis}"}

    # Generate PDF
    pdf_path = generate_pdf_document(
        content=analysis,
        filename=COMPETITOR_ANALYSIS_OUTPUT_PDF,
        title="Competitor Analysis Report (vs X/Twitter)"
    )
    if "Error:" in pdf_path:
         return {"error_message": f"Competitor Analysis PDF generation failed: {pdf_path}"}

    return {"competitor_analysis": analysis, "competitor_pdf_path": pdf_path, "error_message": None}


def generate_email_body_node(state: ReportGenerationState) -> dict:
    """Generates the email body using LLM."""
    print("--- Node: Generating Email Body ---")
    if state.get("error_message"): return {} # Skip if errors occurred

    # Check which reports were generated successfully (by checking PDF paths in state)
    narrative_path = state.get("narrative_pdf_path")
    roadmap_path = state.get("roadmap_pdf_path")
    competitor_path = state.get("competitor_pdf_path")

    reports_generated = []
    if narrative_path and "Error:" not in narrative_path: reports_generated.append(os.path.basename(narrative_path))
    if roadmap_path and "Error:" not in roadmap_path: reports_generated.append(os.path.basename(roadmap_path))
    if competitor_path and "Error:" not in competitor_path: reports_generated.append(os.path.basename(competitor_path))

    if not reports_generated:
        # If no reports generated, set error and skip email body generation
        return {"error_message": "No reports were successfully generated to email."}

    files_list = ", ".join(reports_generated)
    # Generate a dynamic subject line
    subject = f"Bluesky Feedback Analysis Reports ({pd.Timestamp.now().strftime('%Y-%m-%d')})"

    prompt = f"""
You are an assistant composing an email notification.
The following reports have been successfully generated based on recent Bluesky user feedback analysis: {files_list}.

Write a brief, professional email body (2-3 sentences) informing the recipient that the analysis is complete and the reports are attached. Mention the types of reports attached (e.g., Narrative Summary, Prioritized Roadmap, Competitor Analysis).
"""
    email_body = "Error generating email body." # Default
    try:
        response = state["llm_client"].invoke([HumanMessage(content=prompt)])
        email_body = response.content
        print("Email body generated.")
    except Exception as e:
        print(f"Error generating email body: {e}")
        # Fallback to a generic body
        email_body = f"Please find the attached Bluesky feedback analysis reports: {files_list}"

    # Update state with subject and body
    return {"email_subject": subject, "email_body": email_body, "error_message": None}


def send_email_node(state: ReportGenerationState) -> dict:
    """Sends the email with generated PDF attachments."""
    print("--- Node: Sending Email ---")
    if state.get("error_message"):
        final_msg = f"Workflow finished with error before email stage: {state['error_message']}"
        print(final_msg)
        # Store final message in state, even if it's an error from previous step
        return {"final_message": final_msg}

    # Retrieve necessary info from state
    recipient = state.get("recipient_email")
    subject = state.get("email_subject")
    body = state.get("email_body")
    # Get list of generated PDF paths from state
    pdf_paths = [
        state.get("narrative_pdf_path"),
        state.get("roadmap_pdf_path"),
        state.get("competitor_pdf_path")
    ]
    # Filter out None or paths that indicate an error occurred during generation
    valid_pdf_paths = [p for p in pdf_paths if p and isinstance(p, str) and "Error:" not in p]

    # Validate required components for sending email
    if not all([recipient, subject, body]):
        final_msg = "Error: Missing recipient, subject, or body for email."
        print(final_msg)
        return {"error_message": final_msg, "final_message": final_msg}
    if not valid_pdf_paths:
        final_msg = "Error: No valid PDF reports found in state to attach."
        print(final_msg)
        return {"error_message": final_msg, "final_message": final_msg}

    # Call the utility function to send the email
    email_status = send_email_with_attachments(
        recipient_email=recipient,
        subject=subject,
        body=body,
        attachment_paths=valid_pdf_paths
    )

    # Update the final message and error message based on email status
    error_msg = email_status if "Error" in email_status or "Failed" in email_status else None
    return {"final_message": email_status, "error_message": error_msg}


# --- Utility Function to Save Text (can be moved to utils.py) ---
def save_text_to_file(content, filepath):
    """Saves text content to a file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully saved content to {filepath}")
    except Exception as e:
        print(f"Error saving content to {filepath}: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the Bluesky Feedback Reporting Suite.")
    parser.add_argument("recipient_email", help="Email address to send the final reports to.")
    args = parser.parse_args()
    recipient = args.recipient_email # Get email from command line
    print(f"Reports will be sent to: {recipient}")

    print("Starting LangGraph Reporting Suite Runner (with PDF & Email)...")
    load_dotenv()

    # --- Initialize LLM and Tools ---
    llm = None
    openai_api_key = os.getenv('OPENAI_API_KEY')
    # recipient = os.getenv('RECIPIENT_EMAIL') # REMOVED: Get from args instead

    if not recipient: # Should be caught by argparse, but double-check
        exit("Recipient email address is required. Exiting.")

    if openai_api_key:
        try:
            llm = ChatOpenAI(
                temperature=0.3, model_name="gpt-4-turbo-preview",
                openai_api_key=openai_api_key, max_tokens=1000
            )
            print("LLM client initialized successfully.")
        except Exception as e:
            print(f"Error initializing LLM client: {e}")
            # Add fallback if needed
    else:
        print("OpenAI API key not found.")

    search = DuckDuckGoSearchRun()
    print("Search tool initialized.")

    if not llm:
        exit("LLM client failed to initialize. Exiting.")

    # --- Define the Graph ---
    workflow = StateGraph(ReportGenerationState)

    # Add nodes
    workflow.add_node("load_data", load_initial_data_node)
    workflow.add_node("generate_narrative", run_narrative_node)
    workflow.add_node("prioritize_roadmap", run_planner_node)
    workflow.add_node("analyze_competitor", run_competitor_node)
    workflow.add_node("generate_email_body", generate_email_body_node)
    workflow.add_node("send_email", send_email_node)

    # Define the edges
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "generate_narrative")
    workflow.add_edge("generate_narrative", "prioritize_roadmap")
    workflow.add_edge("prioritize_roadmap", "analyze_competitor")
    workflow.add_edge("analyze_competitor", "generate_email_body")
    workflow.add_edge("generate_email_body", "send_email")
    workflow.add_edge("send_email", END)

    # Compile the graph
    app = workflow.compile()
    print("LangGraph compiled successfully.")

    # --- Run the Graph ---
    print("\nInvoking the LangGraph reporting workflow...")
    initial_state = {
        "llm_client": llm,
        "search_tool": search,
        "analyzed_data_path": ANALYZED_DATA_FILE,
        "ranked_topics_path": RANKED_TOPICS_FILE,
        "recipient_email": recipient, # Pass recipient email from args
    }

    final_state = app.invoke(initial_state)

    print("\n--- Workflow Finished ---")
    print("Final State:")
    # Pretty print the final state dictionary, handling potential non-serializable objects
    print(json.dumps(final_state, indent=2, default=lambda x: str(x) if isinstance(x, pd.DataFrame) else repr(x)))

    final_message = final_state.get("final_message", "Workflow finished without a final status message.")
    print(f"\nFinal Status: {final_message}")

    if final_state.get("error_message"):
        print(f"\nWorkflow finished with error: {final_state['error_message']}")
    else:
        print("\nWorkflow completed. Check generated PDF files and email recipient's inbox.")
        # List expected PDF files
        print(f"- {NARRATIVE_OUTPUT_PDF}")
        print(f"- {PRIORITIZED_OUTPUT_PDF}")
        print(f"- {COMPETITOR_ANALYSIS_OUTPUT_PDF}")

