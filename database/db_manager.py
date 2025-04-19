"""
Database manager for the AI-Powered Job Application Screening System.
"""

import logging
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from database.models import Database
import config

logger = logging.getLogger(__name__)

class DBManager:
    """High-level database manager for application-specific operations."""
    
    def __init__(self, db_path=None):
        """Initialize database manager with the specified database path."""
        self.db_path = db_path or config.DB_PATH
        self.db = Database(self.db_path)
    
    def close(self):
        """Close the database connection."""
        self.db.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # Job Description methods
    def import_job_description(self, title, description, skills, qualifications, responsibilities):
        """Import a job description into the database.
        
        Args:
            title (str): Job title
            description (str): Job description
            skills (list): Required skills
            qualifications (list): Required qualifications
            responsibilities (list): Job responsibilities
            
        Returns:
            int: Job ID in the database, or None if import failed
        """
        try:
            # First, check if a job with this title already exists
            all_jobs = self.get_all_jobs()
            existing_job = next((job for job in all_jobs if job['title'].lower() == title.lower()), None)
            
            # If the job already exists, return its ID without creating a duplicate
            if existing_job:
                logger.info(f"Job with title '{title}' already exists with ID {existing_job['id']}")
                return existing_job['id']
            
            # If no existing job was found, proceed with creating a new one
            job_id = self.db.add_job_description(title, description, skills, qualifications, responsibilities)
            logger.info(f"Imported job description: {title} (ID: {job_id})")
            return job_id
                
        except Exception as e:
            logger.error(f"Error importing job description {title}: {str(e)}")
            return None
    
    def get_all_jobs(self):
        """Get all job descriptions."""
        return self.db.get_all_job_descriptions()
    
    def get_job(self, job_id):
        """Get a job description by ID."""
        return self.db.get_job_description(job_id)
    
    def get_job_by_id(self, job_id):
        """Get a job description by ID (alias for get_job)."""
        return self.get_job(job_id)
    
    # Candidate methods
    def import_candidate(self, cv_id, name, email, phone, skills, qualifications, experience, cv_text):
        """Import a candidate into the database."""
        try:
            # Check if candidate already exists
            existing = self.db.get_candidate_by_cv_id(cv_id)
            if existing:
                logger.info(f"Candidate already exists: {cv_id}")
                
                # Update the existing candidate with new data
                self.db.update_candidate(
                    existing['id'],
                    name=name,
                    email=email,
                    phone=phone,
                    skills=skills,
                    qualifications=qualifications,
                    experience=experience,
                    cv_text=cv_text
                )
                logger.info(f"Updated existing candidate: {name} (ID: {existing['id']})")
                return existing['id']
            
            candidate_id = self.db.add_candidate(cv_id, name, email, phone, skills, qualifications, experience, cv_text)
            logger.info(f"Imported candidate: {name} (ID: {candidate_id})")
            return candidate_id
        except Exception as e:
            logger.error(f"Error importing candidate: {str(e)}")
            return None
    
    def get_candidate(self, candidate_id):
        """Get a candidate by ID."""
        return self.db.get_candidate(candidate_id)
    
    def get_candidate_by_cv_id(self, cv_id):
        """Get a candidate by CV ID."""
        return self.db.get_candidate_by_cv_id(cv_id)
    
    # Match Results methods
    def save_match_result(self, job_id, candidate_id, match_score, skills_match, quals_match, exp_match, notes=None):
        """Save a match result between a job and a candidate."""
        try:
            # Automatically shortlist if the match score is above the threshold
            shortlisted = match_score >= config.MATCH_THRESHOLD
            
            # Only save to database if the candidate is shortlisted
            if shortlisted:
                match_id = self.db.add_match_result(
                    job_id, candidate_id, match_score, skills_match, quals_match, exp_match, 
                    shortlisted=shortlisted, notes=notes
                )
                
                logger.info(f"Saved match result: Job {job_id} and Candidate {candidate_id} - " 
                            f"Score: {match_score:.2f}, Shortlisted: {shortlisted}")
                return match_id
            else:
                # Just log the result but don't save to database
                logger.info(f"Match result not saved (below threshold): Job {job_id} and Candidate {candidate_id} - " 
                            f"Score: {match_score:.2f}, Shortlisted: {shortlisted}")
                return None
        except Exception as e:
            logger.error(f"Error saving match result: {str(e)}")
            return None
    
    def get_match_results(self, job_id, shortlisted_only=False):
        """Get match results for a job."""
        return self.db.get_match_results(job_id, shortlisted_only)
    
    def update_shortlist(self, match_id, shortlisted):
        """Update the shortlisting status of a match."""
        return self.db.update_shortlist_status(match_id, shortlisted)
    
    # Interview Request methods
    def create_interview_request(self, match_id, status="pending", scheduled_date=None, 
                                interview_type="video", notes=None):
        """Create an interview request for a match."""
        try:
            interview_id = self.db.add_interview_request(
                match_id, status, scheduled_date, interview_type, notes
            )
            logger.info(f"Created interview request (ID: {interview_id}) for match {match_id}")
            return interview_id
        except Exception as e:
            logger.error(f"Error creating interview request: {str(e)}")
            return None
    
    def mark_email_sent(self, interview_id):
        """Mark an interview request as having had the email sent."""
        try:
            success = self.db.update_interview_email_sent(interview_id)
            if success:
                logger.info(f"Marked interview request {interview_id} as having email sent")
            else:
                logger.warning(f"Failed to update email status for interview request {interview_id}")
            return success
        except Exception as e:
            logger.error(f"Error updating interview email status: {str(e)}")
            return False
    
    def get_pending_interviews(self):
        """Get all pending interview requests that need emails sent."""
        return self.db.get_pending_interviews()
    
    # Shortlisting and Interview Scheduling
    def process_shortlisting(self, job_id):
        """Process shortlisting for a job based on match scores."""
        try:
            # Get all match results for the job
            matches = self.get_match_results(job_id)
            
            # Count how many were shortlisted based on the threshold
            shortlisted_count = sum(1 for match in matches if match['shortlisted'])
            
            logger.info(f"Job {job_id}: {shortlisted_count} candidates shortlisted out of {len(matches)}")
            return shortlisted_count
        except Exception as e:
            logger.error(f"Error processing shortlisting: {str(e)}")
            return 0
    
    def schedule_interviews(self, job_id):
        """Schedule interviews for all shortlisted candidates for a job."""
        try:
            # Get shortlisted candidates
            shortlisted = self.get_match_results(job_id, shortlisted_only=True)
            
            interview_count = 0
            for match in shortlisted:
                # Create an interview request for each shortlisted candidate
                interview_id = self.create_interview_request(match['id'])
                if interview_id:
                    interview_count += 1
            
            logger.info(f"Job {job_id}: Scheduled {interview_count} interviews for shortlisted candidates")
            return interview_count
        except Exception as e:
            logger.error(f"Error scheduling interviews: {str(e)}")
            return 0
            
    def reset_database(self, confirm=False):
        """Clear all data from the database tables.
        
        Args:
            confirm (bool): Confirmation to proceed with reset
            
        Returns:
            bool: True if reset was successful, False otherwise
        """
        if not confirm:
            logger.warning("Database reset was attempted but not confirmed")
            return False
            
        try:
            # Clear all tables in reverse order of dependencies
            self.db.cursor.execute("DELETE FROM interview_requests")
            self.db.cursor.execute("DELETE FROM match_results")
            self.db.cursor.execute("DELETE FROM candidates")
            self.db.cursor.execute("DELETE FROM job_descriptions")
            self.db.conn.commit()
            
            logger.info("Database reset successfully - all data cleared")
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            return False 