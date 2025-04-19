"""
Scheduler Agent

This agent is responsible for sending interview requests to shortlisted candidates.
It generates personalized emails with potential interview dates and times.
"""

import logging
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from database.db_manager import DBManager
from utils.email_generator import EmailGenerator
import config

logger = logging.getLogger(__name__)

class SchedulerAgent:
    """Agent for scheduling interviews with shortlisted candidates."""
    
    def __init__(self):
        """Initialize the Scheduler agent."""
        self.db_manager = DBManager()
        self.email_generator = EmailGenerator()
        self.output_dir = os.path.join(parent_dir, "generated_emails")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def close(self):
        """Close database connections."""
        self.db_manager.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def schedule_interviews_for_job(self, job_id):
        """Schedule interviews for all shortlisted candidates for a job.
        
        Args:
            job_id (int): Job ID in the database
            
        Returns:
            int: Number of interviews scheduled
        """
        # Get job data
        job_data = self.db_manager.get_job(job_id)
        if not job_data:
            logger.warning(f"Job ID {job_id} not found")
            return 0
        
        # Get shortlisted candidates
        match_results = self.db_manager.get_match_results(job_id, shortlisted_only=True)
        
        if not match_results:
            logger.info(f"No shortlisted candidates found for Job {job_id}")
            return 0
        
        logger.info(f"Scheduling interviews for {len(match_results)} shortlisted candidates for Job {job_id}")
        
        # Schedule interviews for each shortlisted candidate
        interview_count = 0
        for match in match_results:
            # Get candidate data
            candidate_data = self.db_manager.get_candidate(match['candidate_id'])
            if not candidate_data:
                continue
            
            # Create an interview request
            interview_id = self.db_manager.create_interview_request(match['id'])
            if not interview_id:
                continue
            
            # Generate interview email
            email_content = self.email_generator.generate_interview_email(
                candidate_data=candidate_data,
                job_data=job_data
            )
            
            # Save email to file
            email_file = self.email_generator.save_email_to_file(
                email_content=email_content,
                candidate_id=candidate_data['id'],
                job_id=job_id,
                output_dir=self.output_dir
            )
            
            # Mark email as sent
            self.db_manager.mark_email_sent(interview_id)
            
            interview_count += 1
        
        logger.info(f"Scheduled {interview_count} interviews for Job {job_id}")
        return interview_count
    
    def schedule_all_pending_interviews(self):
        """Schedule all pending interviews for all jobs.
        
        Returns:
            int: Total number of interviews scheduled
        """
        # Get all jobs
        jobs = self.db_manager.get_all_jobs()
        
        total_interviews = 0
        for job in jobs:
            # First process shortlisting to ensure all candidates are properly flagged
            self.db_manager.process_shortlisting(job['id'])
            
            # Then schedule interviews
            interview_count = self.schedule_interviews_for_job(job['id'])
            total_interviews += interview_count
        
        logger.info(f"Scheduled a total of {total_interviews} interviews across all jobs")
        return total_interviews
    
    def get_interview_status(self, job_id=None):
        """Get status of interview requests.
        
        Args:
            job_id (int, optional): Job ID to filter by
            
        Returns:
            list: List of interview request statuses
        """
        # Get all interview requests
        if job_id:
            query = '''
            SELECT ir.*, mr.job_id, mr.candidate_id, c.name, c.email
            FROM interview_requests ir
            JOIN match_results mr ON ir.match_id = mr.id
            JOIN candidates c ON mr.candidate_id = c.id
            WHERE mr.job_id = ?
            '''
            interviews = self.db_manager.db.cursor.execute(query, (job_id,)).fetchall()
        else:
            query = '''
            SELECT ir.*, mr.job_id, mr.candidate_id, c.name, c.email, j.title as job_title
            FROM interview_requests ir
            JOIN match_results mr ON ir.match_id = mr.id
            JOIN candidates c ON mr.candidate_id = c.id
            JOIN job_descriptions j ON mr.job_id = j.id
            '''
            interviews = self.db_manager.db.cursor.execute(query).fetchall()
        
        # Convert to list of dictionaries
        results = [dict(row) for row in interviews]
        
        return results 