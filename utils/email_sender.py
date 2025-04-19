"""
Email sending utility for the job application screening system.
This module handles sending emails to candidates and storing/retrieving email content.
"""

import os
import smtplib
import logging
import glob
import re
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

# Import config to access predefined email settings
import sys
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
import config

logger = logging.getLogger(__name__)

# Default email configuration
DEFAULT_EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com', 
    'smtp_port': 587,               
    'use_tls': True,
    'username': '',
    'password': '',
    'sender_email': '',
    'sender_name': 'HireIQ System'
}

# Initialize EMAIL_CONFIG with default values
EMAIL_CONFIG = DEFAULT_EMAIL_CONFIG.copy()

# Load email settings from config file if available
if hasattr(config, 'EMAIL_SENDER'):
    EMAIL_CONFIG['sender_email'] = config.EMAIL_SENDER
    EMAIL_CONFIG['username'] = config.EMAIL_SENDER

if hasattr(config, 'PASSWORD'):
    EMAIL_CONFIG['password'] = config.PASSWORD

# Company name from config if available
COMPANY_NAME = getattr(config, 'COMPANY_NAME', 'HireIQ Recruitment Team')

def get_generated_emails_directory() -> Path:
    """Get the path to the generated_emails directory."""
    base_dir = Path(__file__).resolve().parent.parent
    emails_dir = base_dir / "generated_emails"
    os.makedirs(emails_dir, exist_ok=True)
    return emails_dir

def list_all_email_files() -> List[Path]:
    """List all email files in the generated_emails directory."""
    emails_dir = get_generated_emails_directory()
    email_files = list(emails_dir.glob("*.txt"))
    return sorted(email_files)

def get_emails_by_job_id(job_id: int) -> List[Path]:
    """Get all email files for a specific job ID."""
    emails_dir = get_generated_emails_directory()
    email_files = list(emails_dir.glob(f"*job_{job_id}_*.txt"))
    return sorted(email_files)

def get_emails_by_candidate_id(candidate_id: int) -> List[Path]:
    """Get all email files for a specific candidate ID."""
    emails_dir = get_generated_emails_directory()
    email_files = list(emails_dir.glob(f"*candidate_{candidate_id}_*.txt"))
    return sorted(email_files)

def parse_email_file(file_path: Path) -> Dict:
    """Parse an email file to extract subject, recipient, and body."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extract job_id and candidate_id from filename
        job_id = None
        candidate_id = None
        filename = file_path.name
        
        job_match = re.search(r'job_(\d+)', filename)
        if job_match:
            job_id = int(job_match.group(1))
            
        candidate_match = re.search(r'candidate_(\d+)', filename)
        if candidate_match:
            candidate_id = int(candidate_match.group(1))
        
        # Simple parsing: First line is typically the subject
        lines = content.strip().split('\n')
        subject = lines[0].strip()
        
        # Extract recipient if available, otherwise use a default
        recipient = None
        for line in lines[1:5]:  # Check first few lines for email address
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', line)
            if email_match:
                recipient = email_match.group(0)
                break
        
        # Body is everything after the subject
        body = '\n'.join(lines[1:])
        
        return {
            'file_path': str(file_path),
            'job_id': job_id,
            'candidate_id': candidate_id,
            'subject': subject,
            'recipient': recipient,
            'body': body,
            'sent': False  # Default not sent
        }
    except Exception as e:
        logger.error(f"Error parsing email file {file_path}: {str(e)}")
        return {
            'file_path': str(file_path),
            'error': str(e)
        }

def get_all_emails_data() -> List[Dict]:
    """Get data for all email files."""
    email_files = list_all_email_files()
    emails_data = []
    
    for file_path in email_files:
        email_data = parse_email_file(file_path)
        emails_data.append(email_data)
    
    return emails_data

def send_email(recipient: str, subject: str, body: str, html_body: Optional[str] = None) -> bool:
    """Send an email using SMTP."""
    if not recipient:
        logger.error("No recipient specified for email")
        return False
    
    # Use config values if they exist
    username = EMAIL_CONFIG['username']
    password = EMAIL_CONFIG['password']
    
    if not username or not password:
        logger.error("Email credentials not configured. Check your config file for EMAIL_SENDER and PASSWORD settings.")
        return False
    
    # Create the email
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['sender_email']}>"
    msg['To'] = recipient
    
    # Attach text part
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach HTML part if provided
    if html_body:
        msg.attach(MIMEText(html_body, 'html'))
    
    try:
        # Connect to the SMTP server and send the email
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            if EMAIL_CONFIG['use_tls']:
                server.starttls()
            
            server.login(username, password)
            server.send_message(msg)
        
        logger.info(f"Email sent successfully to {recipient}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {recipient}: {str(e)}")
        return False

def send_emails_to_candidates(email_ids: List[int] = None) -> Dict:
    """Send emails to candidates based on the selected email IDs."""
    all_emails = get_all_emails_data()
    
    # If email_ids is provided, filter the emails
    if email_ids:
        emails_to_send = [email for i, email in enumerate(all_emails) if i in email_ids]
    else:
        emails_to_send = all_emails
    
    results = {
        'total': len(emails_to_send),
        'sent': 0,
        'failed': 0,
        'details': []
    }
    
    for email in emails_to_send:
        if 'error' in email or not email.get('recipient'):
            results['failed'] += 1
            results['details'].append({
                'status': 'failed',
                'reason': email.get('error', 'No recipient specified'),
                'file': email.get('file_path')
            })
            continue
        
        success = send_email(
            recipient=email['recipient'],
            subject=email['subject'],
            body=email['body']
        )
        
        if success:
            results['sent'] += 1
            results['details'].append({
                'status': 'sent',
                'file': email['file_path'],
                'recipient': email['recipient']
            })
        else:
            results['failed'] += 1
            results['details'].append({
                'status': 'failed',
                'reason': 'SMTP error',
                'file': email['file_path'],
                'recipient': email['recipient']
            })
    
    return results

def save_email_configuration(config: Dict) -> bool:
    """Save email configuration to a file."""
    try:
        config_dir = Path(__file__).resolve().parent.parent / "config"
        os.makedirs(config_dir, exist_ok=True)
        
        config_file = config_dir / "email_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update the global EMAIL_CONFIG
        global EMAIL_CONFIG
        EMAIL_CONFIG.update(config)
        
        return True
    except Exception as e:
        logger.error(f"Error saving email configuration: {str(e)}")
        return False

def load_email_configuration() -> Dict:
    """Load email configuration from a file."""
    try:
        config_dir = Path(__file__).resolve().parent.parent / "config"
        config_file = config_dir / "email_config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update the global EMAIL_CONFIG
            global EMAIL_CONFIG
            EMAIL_CONFIG.update(config_data)
        
        # If there are config values, make sure they're included
        if hasattr(config, 'EMAIL_SENDER'):
            EMAIL_CONFIG['sender_email'] = config.EMAIL_SENDER
            EMAIL_CONFIG['username'] = config.EMAIL_SENDER
            
        if hasattr(config, 'PASSWORD'):
            EMAIL_CONFIG['password'] = config.PASSWORD
            
        return EMAIL_CONFIG
    except Exception as e:
        logger.error(f"Error loading email configuration: {str(e)}")
        return EMAIL_CONFIG 