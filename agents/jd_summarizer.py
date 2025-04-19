"""
Job Description Summarizer Agent

This agent reads job descriptions and extracts key elements:
- Required skills
- Qualifications
- Responsibilities
"""

import logging
import csv
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from database.db_manager import DBManager
from utils.text_processor import TextProcessor
import config

logger = logging.getLogger(__name__)

class JDSummarizerAgent:
    """Agent for summarizing job descriptions."""
    
    def __init__(self):
        """Initialize the JD Summarizer agent."""
        self.text_processor = TextProcessor()
        self.db_manager = DBManager()
    
    def close(self):
        """Close database connections."""
        self.db_manager.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def process_job_description(self, title, description):
        """Process a single job description.
        
        Args:
            title (str): Job title
            description (str): Job description text
            
        Returns:
            int: Job ID in the database
        """
        # Extract key elements from the job description
        job_data = self.text_processor.summarize_job_description(title, description)
        
        logger.info(f"Processed job description: {title}")
        logger.info(f"Extracted {len(job_data['skills'])} skills, "
                   f"{len(job_data['qualifications'])} qualifications, "
                   f"{len(job_data['responsibilities'])} responsibilities")
        
        # Store the processed job in the database
        job_id = self.db_manager.import_job_description(
            title=job_data['title'],
            description=job_data['description'],
            skills=job_data['skills'],
            qualifications=job_data['qualifications'],
            responsibilities=job_data['responsibilities']
        )
        
        return job_id
    
    def process_job_descriptions_from_csv(self, csv_path=None):
        """Process job descriptions from a CSV file.
        
        Args:
            csv_path (str, optional): Path to the CSV file containing job descriptions
            
        Returns:
            list: List of processed job IDs
        """
        if csv_path is None:
            csv_path = config.JD_CSV_PATH
        
        job_ids = []
        
        try:
            with open(csv_path, 'r', encoding='windows-1252') as csvfile:
                csv_reader = csv.reader(csvfile)
                
                # Skip header row
                header = next(csv_reader, None)
                
                if not header or len(header) < 2:
                    logger.error(f"Invalid CSV format in {csv_path}")
                    return job_ids
                
                for row in csv_reader:
                    if len(row) < 2 or not row[0] or not row[1]:
                        continue  # Skip empty rows
                    
                    title = row[0].strip()
                    description = row[1].strip()
                    
                    job_id = self.process_job_description(title, description)
                    if job_id:
                        job_ids.append(job_id)
            
            logger.info(f"Processed {len(job_ids)} job descriptions from {csv_path}")
        
        except Exception as e:
            logger.error(f"Error processing job descriptions from CSV: {str(e)}")
        
        return job_ids
    
    def get_job_summary(self, job_id):
        """Get a summary of a job description.
        
        Args:
            job_id (int): Job ID in the database
            
        Returns:
            dict: Job summary
        """
        job_data = self.db_manager.get_job(job_id)
        
        if not job_data:
            logger.warning(f"Job ID {job_id} not found")
            return None
        
        # Create a formatted summary
        summary = {
            'title': job_data['title'],
            'skills': job_data['skills'],
            'qualifications': job_data['qualifications'],
            'responsibilities': job_data['responsibilities']
        }
        
        return summary
    
    def get_all_job_summaries(self):
        """Get summaries of all job descriptions.
        
        Returns:
            list: List of job summaries
        """
        jobs = self.db_manager.get_all_jobs()
        
        summaries = []
        for job in jobs:
            summary = {
                'id': job['id'],
                'title': job['title'],
                'skills': job['skills'],
                'qualifications': job['qualifications'],
                'responsibilities': job['responsibilities']
            }
            summaries.append(summary)
        
        return summaries 