import logging
import smtplib
from email.message import EmailMessage
from pathlib import Path
import os
from src.config.settings import Config

# Sample SMTP configuration
SMTP_SERVER = Config.SMTP_SERVER
SMTP_PORT = Config.SMTP_PORT

SMTP_USER = Config.SMTP_USER
SMTP_PASSWORD = Config.SMTP_PASSWORD

logger = logging.getLogger(__name__)

def send_email(to_addresses: list, subject: str, body: str, attachments: list = None):
    if not SMTP_USER or not SMTP_PASSWORD:
        return {"status": "failed", "error": "SMTP credentials are not configured."}
    if not to_addresses:
        return {"status": "failed", "error": "No recipient addresses provided."}

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = ", ".join(to_addresses)
    msg["Subject"] = subject
    msg.set_content(body)

    attachments = attachments or []

    try:
        for file_path in attachments:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Attachment not found: {file_path}")

            with open(path, "rb") as f:
                file_data = f.read()

            msg.add_attachment(
                file_data,
                maintype="application",
                subtype="octet-stream",
                filename=path.name
            )

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return {"status": "failed", "error": str(e)}

    return {"status": "sent", "recipients": to_addresses, "subject": subject}