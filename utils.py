import os
import textwrap
from fpdf import FPDF # Requires: pip install fpdf2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# --- PDF Generation ---

def generate_pdf_document(content: str, filename: str, title: str = "Report"):
    """
    Generates a PDF document from the given text content.

    Args:
        content: The text content to include in the PDF.
        filename: The name of the PDF file to save (e.g., "report.pdf").
        title: An optional title for the document header.

    Returns:
        str: The path to the generated PDF file if successful, otherwise an error message.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Set up fonts
    try:
        pdf.set_font("Arial", "B", 16)
    except RuntimeError:
        print("Arial font not found, using default. Install Arial if needed.")
        pdf.set_font("Helvetica", "B", 16) # Fallback font

    pdf.cell(0, 10, title, 0, 1, "C") # Add title centered
    pdf.ln(10) # Add space after title

    try:
        pdf.set_font("Arial", "", 11) # Slightly smaller font for body
    except RuntimeError:
        pdf.set_font("Helvetica", "", 11)

    # Add content, handling line breaks and wrapping
    lines = content.split('\n')
    for line in lines:
        # Use textwrap to handle lines that are too wide for the PDF page
        # Adjust width based on standard A4 page width minus margins
        wrapped_lines = textwrap.wrap(line, width=85) # Adjust width as needed

        if not wrapped_lines: # Handle empty lines
             pdf.ln(5) # Add some vertical space for empty lines
        else:
            for wrapped_line in wrapped_lines:
                # Encode properly for PDF
                try:
                    pdf.multi_cell(0, 5, wrapped_line.encode('latin-1', 'replace').decode('latin-1')) # Use multi_cell for auto line breaks within cell
                except UnicodeEncodeError:
                     # Fallback for characters not in latin-1
                     pdf.multi_cell(0, 5, wrapped_line.encode('ascii', 'replace').decode('ascii'))
                # Removed the extra pdf.ln() as multi_cell handles vertical spacing

    # Save the PDF
    try:
        pdf.output(filename)
        print(f"Generated PDF: {filename}")
        # Return the filename so the agent knows the file was created
        return filename
    except Exception as e:
        print(f"Error saving PDF {filename}: {e}")
        return f"Error: Failed to save PDF {filename}"

# --- Email Sending ---

def send_email_with_attachments(recipient_email: str, subject: str, body: str, attachment_paths: list):
    """
    Sends an email with attachments using credentials from environment variables.

    Args:
        recipient_email: The email address of the recipient.
        subject: The subject line of the email.
        body: The body text of the email (plain text).
        attachment_paths: A list of file paths to attach.

    Returns:
        str: A success or error message string.
    """
    sender_email = os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_PASSWORD") # Often an App Password for Gmail/Outlook
    smtp_server = os.getenv("SMTP_SERVER") # e.g., smtp.gmail.com
    smtp_port = int(os.getenv("SMTP_PORT", 587)) # Default to 587 (TLS)

    if not all([sender_email, sender_password, smtp_server, recipient_email]):
        error_msg = "Error: Email configuration missing in environment variables (EMAIL_ADDRESS, EMAIL_PASSWORD, SMTP_SERVER, RECIPIENT_EMAIL)."
        print(error_msg)
        return error_msg

    print(f"Preparing email to {recipient_email} from {sender_email} via {smtp_server}:{smtp_port}")

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # Attach files
    valid_attachments = []
    for file_path in attachment_paths:
        if not file_path or not isinstance(file_path, str):
             print(f"Warning: Invalid attachment path skipped: {file_path}")
             continue
        try:
            with open(file_path, "rb") as f:
                part = MIMEApplication(f.read(), name=os.path.basename(file_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            msg.attach(part)
            valid_attachments.append(os.path.basename(file_path))
            print(f"Attached: {os.path.basename(file_path)}")
        except FileNotFoundError:
            error_msg = f"Error: Attachment not found at {file_path}"
            print(error_msg)
            return error_msg # Fail fast if attachment missing
        except Exception as e:
            error_msg = f"Error attaching file {file_path}: {e}"
            print(error_msg)
            return error_msg # Fail fast

    if not valid_attachments:
        error_msg = "Error: No valid attachments found or provided to send."
        print(error_msg)
        return error_msg

    # Send the email
    try:
        print(f"Connecting to SMTP server {smtp_server}...")
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo() # Identify ourselves to the server
            server.starttls() # Secure the connection
            server.ehlo() # Re-identify ourselves over TLS
            print("Logging into SMTP server...")
            server.login(sender_email, sender_password)
            print("Sending email...")
            text = msg.as_string()
            server.sendmail(sender_email, recipient_email, text)
        success_msg = f"Email with attachments ({', '.join(valid_attachments)}) sent successfully to {recipient_email}"
        print(success_msg)
        return success_msg

    except smtplib.SMTPAuthenticationError:
        error_msg = "Email sending failed (SMTP Authentication Error). Check EMAIL_ADDRESS and EMAIL_PASSWORD (use App Password if needed)."
        print(error_msg)
        return error_msg
    except smtplib.SMTPConnectError:
        error_msg = f"Email sending failed (SMTP Connection Error). Could not connect to {smtp_server}:{smtp_port}."
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Failed to send email: {e}"
        print(error_msg)
        return error_msg

