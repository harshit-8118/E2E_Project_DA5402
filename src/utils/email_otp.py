# src/utils/email_otp.py
# Gmail SMTP OTP sender for email verification
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[2] / ".env",
    override=True
)

import os
import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from src.utils.logger import get_logger


logger = get_logger("email_otp")

GMAIL_USER     = os.getenv("SMTP_USERNAME",     "")
GMAIL_APP_PASS = os.getenv("SMTP_AUTH_PASSWORD", "")


def generate_otp(length: int = 6) -> str:
    return "".join([str(random.randint(0, 9)) for _ in range(length)])


def send_otp_email(to_email: str, otp: str, username: str) -> bool:
    """Send OTP verification email via Gmail SMTP. Returns True on success."""
    if not GMAIL_USER or not GMAIL_APP_PASS:
        logger.warning("Gmail credentials not set — OTP email skipped (dev mode)")
        logger.info(f"[DEV MODE] OTP for {to_email}: {otp}")
        return True   # allow dev testing without email

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "DermAI — Verify Your Email"
        msg["From"]    = GMAIL_USER
        msg["To"]      = to_email

        html = f"""
        <html><body style="font-family:sans-serif;background:#0a0a0f;color:#e8e8f0;padding:32px">
          <div style="max-width:480px;margin:auto;background:#111118;border:1px solid #2a2a3a;border-radius:16px;padding:32px">
            <h2 style="color:#6ee7b7;margin-bottom:8px">DermAI</h2>
            <p style="color:#6b6b80;font-size:13px;margin-bottom:24px">Skin Disease Detection System</p>
            <p>Hi <strong>{username}</strong>,</p>
            <p>Your email verification code is:</p>
            <div style="background:#1a1a24;border:1px solid #6ee7b7;border-radius:12px;padding:24px;text-align:center;margin:24px 0">
              <span style="font-size:36px;font-weight:800;letter-spacing:8px;color:#6ee7b7">{otp}</span>
            </div>
            <p style="color:#6b6b80;font-size:13px">This code expires in <strong>10 minutes</strong>.</p>
            <p style="color:#6b6b80;font-size:13px">If you did not create a DermAI account, ignore this email.</p>
            <hr style="border-color:#2a2a3a;margin:24px 0"/>
            <p style="color:#6b6b80;font-size:12px">DermAI — DA5402 MLOps | IIT Madras<br/>
            Not a substitute for professional medical diagnosis.</p>
          </div>
        </body></html>
        """

        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PASS)
            server.sendmail(GMAIL_USER, to_email, msg.as_string())

        logger.info(f"OTP email sent to {to_email}")
        return True

    except Exception as e:
        logger.error(f"OTP email failed for {to_email}: {e}")
        return False