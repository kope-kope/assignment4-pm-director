import os
import subprocess
import sys
from flask import Flask, request, jsonify, render_template
import threading
import time # Needed for sleep if checking thread status

# --- Flask App Initialization ---
# Ensure the static folder serves agent_ui.html if it's not in templates
# If agent_ui.html is in root, use:
# app = Flask(__name__, static_folder='.', static_url_path='')
# If agent_ui.html is in templates, use:
app = Flask(__name__)


# --- Configuration ---
MAIN_SCRIPT_PATH = "main.py"
REPORTING_SCRIPT_PATH = "run_report_graph.py"
# Define expected output files from the reporting script
NARRATIVE_OUTPUT_PDF = "narrative_summary.pdf"
PRIORITIZED_OUTPUT_PDF = "prioritized_roadmap.pdf"
COMPETITOR_ANALYSIS_OUTPUT_PDF = "competitor_analysis_report.pdf"


# Store status of background tasks (more detailed)
# Using a lock for thread safety when accessing/modifying status
status_lock = threading.Lock()
tasks_status = {
    "analysis": {"status": "idle", "message": "Ready"}, # idle, running, completed, error
    "reporting": {"status": "idle", "message": "Ready"}  # idle, running, generating_narrative, prioritizing, analyzing_competitor, sending_email, completed, error
}
# Store background threads to check if they are alive (optional)
background_threads = {}

# --- Helper Function to Run Scripts in Background ---
def run_script_background(script_path, args_list=None, task_key=None):
    """Runs a python script in a separate thread and updates status."""
    global tasks_status
    python_executable = sys.executable
    command = [python_executable, script_path]
    if args_list:
        command.extend(args_list)

    print(f"Starting background task '{task_key}': {' '.join(command)}")
    with status_lock:
        tasks_status[task_key] = {"status": "running", "message": "Process started..."}

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate() # Wait for completion

        # --- Log Output ---
        print(f"--- START {os.path.basename(script_path)} Output ---")
        if stdout: print(f"STDOUT:\n{stdout}")
        if stderr: print(f"STDERR:\n{stderr}")
        print(f"--- END {os.path.basename(script_path)} Output ---")
        # --- End Log Output ---

        final_status = "completed" if process.returncode == 0 else "error"
        final_message = f"Finished successfully." if final_status == "completed" else f"Failed with return code {process.returncode}. Check logs."

        print(f"Background task '{task_key}' finished with status: {final_status}")

    except FileNotFoundError:
        final_status = "error"
        final_message = f"Script not found at {script_path}"
        print(f"Error for task '{task_key}': {final_message}")
    except Exception as e:
        final_status = "error"
        final_message = f"An exception occurred: {e}"
        print(f"Error for task '{task_key}': {final_message}")

    with status_lock:
        tasks_status[task_key] = {"status": final_status, "message": final_message}

    # Clean up thread reference (optional)
    if task_key in background_threads:
        del background_threads[task_key]


# --- API Endpoints ---

@app.route('/')
def index():
    """Serves the main HTML UI file."""
    # Assumes agent_ui.html is in the 'templates' folder
    return render_template('agent_ui.html')

@app.route('/run-analysis', methods=['POST'])
def trigger_analysis():
    """Triggers the main.py script."""
    task_key = "analysis"
    with status_lock:
        if tasks_status[task_key]["status"] == "running":
             return jsonify({"status": "error", "message": "Analysis is already running."}), 409

    print("Received request to run analysis...")
    thread = threading.Thread(target=run_script_background, args=(MAIN_SCRIPT_PATH,), kwargs={"task_key": task_key}, daemon=True)
    background_threads[task_key] = thread # Store thread reference
    thread.start()

    return jsonify({"status": "success", "message": "Analysis script started in background."})

@app.route('/run-reporting', methods=['POST'])
def trigger_reporting():
    """Triggers the run_report_graph.py script with recipient email."""
    task_key = "reporting"
    with status_lock:
        if tasks_status[task_key]["status"] == "running":
             return jsonify({"status": "error", "message": "Reporting is already running."}), 409

    data = request.get_json()
    recipient_email = data.get('recipient_email')

    if not recipient_email:
        return jsonify({"status": "error", "message": "Recipient email is required."}), 400
    if '@' not in recipient_email or '.' not in recipient_email.split('@')[-1]:
         return jsonify({"status": "error", "message": "Invalid email format provided."}), 400

    print(f"Received request to run reporting for: {recipient_email}")
    thread = threading.Thread(target=run_script_background, args=(REPORTING_SCRIPT_PATH, [recipient_email]), kwargs={"task_key": task_key}, daemon=True)
    background_threads[task_key] = thread # Store thread reference
    thread.start()

    return jsonify({"status": "success", "message": f"Reporting script started for {recipient_email} in background."})

@app.route('/status', methods=['GET'])
def get_status():
    """Returns the status, inferring reporting stage based on file existence."""
    global tasks_status
    with status_lock:
        # Create a copy to avoid modifying the original dict while iterating/checking
        current_status = tasks_status.copy()

        # Infer reporting stage if it's running
        if current_status["reporting"]["status"] == "running":
            # Check for output files to determine progress
            try:
                # Check in reverse order of creation
                if os.path.exists(COMPETITOR_ANALYSIS_OUTPUT_PDF):
                    current_status["reporting"]["message"] = "Sending Email..."
                elif os.path.exists(PRIORITIZED_OUTPUT_PDF):
                    current_status["reporting"]["message"] = "Analyzing Competitor..."
                elif os.path.exists(NARRATIVE_OUTPUT_PDF):
                    current_status["reporting"]["message"] = "Prioritizing Roadmap..."
                else:
                    # If none of the output files exist yet, it's likely generating the narrative
                    current_status["reporting"]["message"] = "Generating Narrative Summary..."
            except Exception as e:
                 print(f"Error checking file existence for status: {e}")
                 # Keep the last known message if file check fails
                 current_status["reporting"]["message"] = tasks_status["reporting"].get("message", "Running...")


    return jsonify(current_status)


# --- Run the Flask App ---
if __name__ == '__main__':
    # Get port from environment variable for Render, default to 5001 locally
    port = int(os.environ.get('PORT', 5001))
    # Use host='0.0.0.0' for deployment
    app.run(host='0.0.0.0', port=port, debug=False) # Turn Debug OFF for deployment) # Changed debug to False for cleaner logs

