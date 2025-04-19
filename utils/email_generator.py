"""
Email generator utility for the AI-Powered Job Application Screening System.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys
import random
import string

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config

logger = logging.getLogger(__name__)

class EmailGenerator:
    """Utility for generating interview request emails."""
    
    def __init__(self, template_dir=None):
        """Initialize the email generator.
        
        Args:
            template_dir (str, optional): Directory containing email templates.
        """
        self.template_dir = template_dir or config.EMAIL_TEMPLATE_PATH
        self.sender = config.EMAIL_SENDER
        
        # Create templates directory if it doesn't exist
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Create default template if it doesn't exist
        self.default_template_path = os.path.join(self.template_dir, "interview_request.txt")
        if not os.path.exists(self.default_template_path):
            self._create_default_template()
    
    def _create_default_template(self):
        """Create a default email template."""
        template = """Subject: Interview Invitation for {job_title} Position

Dear {candidate_name},

We are pleased to inform you that your application for the {job_title} position has been shortlisted. Your qualifications and experience align well with what we're looking for in this role.

We would like to invite you to an interview to discuss your application further. The details are as follows:

Interview Type: {interview_type}
Date: {interview_date}
Time: {interview_time}

{additional_instructions}

Please confirm your availability by replying to this email. If the proposed time doesn't work for you, please suggest alternative times that would be more convenient.

We look forward to speaking with you and learning more about your skills and experience.

Best regards,
Recruitment Team
{company_name}
{contact_info}
"""
        
        with open(self.default_template_path, 'w', encoding='windows-1252') as f:
            f.write(template)
    
    def generate_interview_dates(self, num_dates=3, start_date=None):
        """Generate potential interview dates.
        
        Args:
            num_dates (int): Number of interview dates to generate
            start_date (datetime, optional): Starting date for interviews
            
        Returns:
            list: List of interview date strings
        """
        if start_date is None:
            # Start from tomorrow
            start_date = datetime.now() + timedelta(days=1)
        
        # Skip weekends
        if start_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            start_date += timedelta(days=(7 - start_date.weekday()))
        
        dates = []
        current_date = start_date
        
        while len(dates) < num_dates:
            # Skip weekends
            if current_date.weekday() < 5:  # 0-4 = Monday-Friday
                dates.append(current_date.strftime('%Y-%m-%d'))
            
            current_date += timedelta(days=1)
        
        return dates
    
    def generate_interview_times(self, num_times=3, start_hour=9, end_hour=16):
        """Generate potential interview times.
        
        Args:
            num_times (int): Number of interview times to generate
            start_hour (int): Earliest hour for interviews (24-hour format)
            end_hour (int): Latest hour for interviews (24-hour format)
            
        Returns:
            list: List of interview time strings
        """
        times = []
        hours = list(range(start_hour, end_hour + 1))
        minutes = [0, 30]  # 30-minute intervals
        
        all_times = []
        for hour in hours:
            for minute in minutes:
                if hour == end_hour and minute > 0:
                    continue  # Skip times past end_hour
                
                time_str = f"{hour:02d}:{minute:02d}"
                all_times.append(time_str)
        
        # Randomly select times
        selected_times = random.sample(all_times, min(num_times, len(all_times)))
        
        # Sort times
        selected_times.sort()
        
        return selected_times
    
    def generate_interview_email(self, candidate_data, job_data, interview_data=None):
        """Generate an interview request email.
        
        Args:
            candidate_data (dict): Candidate information
            job_data (dict): Job information
            interview_data (dict, optional): Interview details
            
        Returns:
            str: Generated email content
        """
        # Set default interview data if not provided
        if interview_data is None:
            interview_data = {}
        
        # Generate interview dates and times if not provided
        if 'dates' not in interview_data:
            interview_data['dates'] = self.generate_interview_dates()
        
        if 'times' not in interview_data:
            interview_data['times'] = self.generate_interview_times()
        
        # Set default interview type if not provided
        if 'type' not in interview_data:
            interview_data['type'] = "Video Interview"
        
        # Load email template
        with open(self.default_template_path, 'r', encoding='windows-1252') as f:
            template = f.read()
        
        # Format dates and times for the email
        date_options = '\n'.join([f"- {date}" for date in interview_data['dates']])
        time_options = '\n'.join([f"- {time}" for time in interview_data['times']])
        
        # Additional instructions based on interview type
        if interview_data.get('type', '').lower() == 'video':
            additional_instructions = """
We will send you a link to join the video call after you confirm your preferred date and time.
Please ensure you have a stable internet connection and a quiet environment for the interview.
"""
        elif interview_data.get('type', '').lower() == 'phone':
            additional_instructions = """
We will call you at the phone number provided in your application.
Please ensure you are available and in a quiet environment for the interview.
"""
        else:
            additional_instructions = """
Further details about the interview will be provided once you confirm your availability.
"""
        
        # Company information
        company_name = "Our Company"  # Placeholder - would come from configuration
        contact_info = "recruitment@example.com | (555) 123-4567"  # Placeholder
        
        # Format the email
        email_content = template.format(
            job_title=job_data.get('title', 'Open Position'),
            candidate_name=candidate_data.get('name', 'Candidate'),
            email=candidate_data.get('email', 'candidate@example.com'),
            interview_type=interview_data.get('type', 'Interview'),
            interview_date=f"Please select from the following dates:\n{date_options}",
            interview_time=f"Please select from the following times:\n{time_options}",
            additional_instructions=additional_instructions,
            company_name=company_name,
            contact_info=contact_info
        )
        
        return email_content
    
    def save_email_to_file(self, email_content, candidate_id, job_id, output_dir=None):
        """Save generated email to a file.
        
        Args:
            email_content (str): Email content
            candidate_id (str): Candidate ID
            job_id (str): Job ID
            output_dir (str, optional): Directory to save the email
            
        Returns:
            str: Path to the saved email file
        """
        if output_dir is None:
            output_dir = os.path.join(parent_dir, "generated_emails")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        filename = f"interview_request_job_{job_id}_candidate_{candidate_id}_{timestamp}_{random_str}.txt"
        
        file_path = os.path.join(output_dir, filename)
        
        # Save email to file
        with open(file_path, 'w', encoding='windows-1252') as f:
            f.write(email_content)
        
        logger.info(f"Saved interview request email to {file_path}")
        
        return file_path 