"""
Automated Email Alert Service for MLOps Data Drift Monitoring
Sends notifications via SMTP (e.g. Gmail) when HIGH drift is detected.
Includes cooldown logic to prevent duplicate spam and graceful error handling.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load credentials from .env if present
load_dotenv()


class EmailAlertService:
    def __init__(self, cooldown_minutes: int = 30):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.sender_email = os.getenv("SMTP_SENDER_EMAIL", "")
        self.app_password = os.getenv("SMTP_APP_PASSWORD", "")
        self.receiver_email = os.getenv("SMTP_RECEIVER_EMAIL", "")
        self.cooldown_minutes = cooldown_minutes
        self.last_alert_time = None
        self.last_alert_features = []

    def is_configured(self) -> bool:
        """
        Checks if SMTP credentials are provided in the environment.
        """
        return bool(self.sender_email and self.app_password and self.receiver_email)

    def should_suppress_alert(self, high_drift_features: List[str]) -> bool:
        """
        Checks if an alert for the same features was recently sent to prevent spam.
        """
        if self.last_alert_time is None:
            return False

        elapsed = datetime.now() - self.last_alert_time
        if elapsed < timedelta(minutes=self.cooldown_minutes):
            if set(high_drift_features) == set(self.last_alert_features):
                return True
        return False

    def send_drift_alert(self, drift_results: Dict[str, Any]) -> bool:
        """
        Sends an HTML email alert when HIGH drift is detected.
        """
        severity = drift_results.get("overall_severity", "UNKNOWN")
        high_features = drift_results.get("high_drift_features", [])
        report_path = drift_results.get("report_filepath", "N/A")
        sample_size = drift_results.get("production_sample_size", 0)

        # Only alert on HIGH DRIFT
        if severity != "HIGH DRIFT":
            print(f"[INFO] Alert bypassed: Drift severity is '{severity}' (Alerts trigger only on HIGH DRIFT).")
            return False

        # Check cooldown
        if self.should_suppress_alert(high_features):
            print(f"[INFO] Alert suppressed: Cooldown active ({self.cooldown_minutes} min threshold).")
            return False

        # Verify configuration
        if not self.is_configured():
            print(
                "[WARNING] Email alert skipped: SMTP credentials not set in environment or .env file.\n"
                "To enable emails, add SMTP_SENDER_EMAIL, SMTP_APP_PASSWORD, and SMTP_RECEIVER_EMAIL to your .env file."
            )
            return False

        # Compose HTML email
        subject = f"🚨 [MLOps ALERT] HIGH Data Drift Detected - Google Play Store Rating Model"

        features_html = "".join([f"<li><b>{feat}</b></li>" for feat in high_features])

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="background-color: #d9534f; color: white; padding: 15px; border-radius: 5px;">
                <h2 style="margin: 0;">⚠️ MLOps Production Alert: HIGH Data Drift Detected</h2>
            </div>
            <p>The automated data drift monitor has detected statistically significant drift in live Google Play Store predictions.</p>
            
            <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Metric</th>
                    <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Value</th>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Drift Severity</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd; color: #d9534f; font-weight: bold;">HIGH DRIFT</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Production Sample Size</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{sample_size} predictions</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Timestamp</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Detailed Report</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><code>{report_path}</code></td>
                </tr>
            </table>

            <h3>Critical Features Affected (PSI &ge; 0.25):</h3>
            <ul>
                {features_html}
            </ul>

            <p style="background-color: #fff3cd; padding: 10px; border-left: 5px solid #ffeeba;">
                <b>Recommended Action:</b> Inspect recent incoming payload patterns. Consider scheduling a model retraining pipeline run if live distributions have permanently shifted.
            </p>
            <p style="font-size: 12px; color: #777;">Google Play Store Rating Prediction MLOps Pipeline &bull; Automated Alert System</p>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = self.receiver_email
        msg.attach(MIMEText(html_body, "html"))

        try:
            print(f"[INFO] Connecting to SMTP server {self.smtp_server}:{self.smtp_port}...")
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10)
            server.starttls()
            server.login(self.sender_email, self.app_password)
            server.sendmail(self.sender_email, [self.receiver_email], msg.as_string())
            server.quit()

            self.last_alert_time = datetime.now()
            self.last_alert_features = list(high_features)
            print(f"[SUCCESS] Email alert dispatched to: {self.receiver_email}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to send email alert: {str(e)}")
            # Do NOT crash monitoring pipeline on email failures
            return False


# Global alert instance
email_alert_service = EmailAlertService()
