"""
CV Processor Agent

This agent extracts key data from CVs including:
- Personal information
- Skills
- Education/Qualifications
- Work Experience
"""

import logging
import os
import glob
from pathlib import Path
import sys
import traceback

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from database.db_manager import DBManager
from utils.pdf_extractor import PDFExtractor
import config

logger = logging.getLogger(__name__)

class CVProcessorAgent:
    """Agent for processing CV files and extracting relevant data."""
    
    def __init__(self):
        """Initialize the CV Processor agent."""
        self.pdf_extractor = PDFExtractor()
        self.db_manager = DBManager()
    
    def close(self):
        """Close database connections."""
        self.db_manager.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def process_cv_file(self, cv_path):
        """Process a single CV file.
        
        Args:
            cv_path (str): Path to the CV file
            
        Returns:
            int: Candidate ID in the database
        """
        try:
            # Check if file exists and is readable
            if not os.path.isfile(cv_path):
                logger.error(f"File does not exist: {cv_path}")
                return None
                
            # Get file size for debugging
            file_size = os.path.getsize(cv_path)
            logger.info(f"Processing CV file: {cv_path} (Size: {file_size} bytes)")
            
            # Extract data from CV
            cv_data = self.pdf_extractor.extract_cv_data(cv_path)
            
            # Store the extracted data in the database
            candidate_id = self.db_manager.import_candidate(
                cv_id=cv_data['cv_id'],
                name=cv_data['name'],
                email=cv_data['email'],
                phone=cv_data['phone'],
                skills=cv_data['skills'],
                qualifications=cv_data['qualifications'],
                experience=cv_data['experience'],
                cv_text=cv_data['cv_text']
            )
            
            logger.info(f"Processed CV: {cv_data['cv_id']} - {cv_data['name']}")
            return candidate_id
        
        except Exception as e:
            logger.error(f"Error processing CV {cv_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def process_cv_directory(self, cv_dir=None):
        """Process all CV files in a directory.
        
        Args:
            cv_dir (str, optional): Directory containing CV files
            
        Returns:
            list: List of processed candidate IDs
        """
        if cv_dir is None:
            cv_dir = config.CV_DIR
        
        candidate_ids = []
        
        try:
            # Check if directory exists
            if not os.path.isdir(cv_dir):
                logger.error(f"Directory does not exist: {cv_dir}")
                return candidate_ids
                
            # Log the absolute path for debugging
            abs_path = os.path.abspath(cv_dir)
            logger.info(f"Looking for CVs in directory: {abs_path}")
            
            # List all files in the directory for debugging
            all_files = os.listdir(cv_dir)
            pdf_files_in_dir = [f for f in all_files if f.lower().endswith('.pdf')]
            logger.info(f"Directory contains {len(all_files)} files, {len(pdf_files_in_dir)} PDF files")
            
            # Find all PDF files in the directory using os.listdir() instead of glob
            # This avoids glob pattern issues on Windows
            pdf_files = [os.path.join(cv_dir, f) for f in all_files if f.lower().endswith('.pdf')]
            
            logger.info(f"Found {len(pdf_files)} CV files in {cv_dir}")
            
            # Process each CV file
            for pdf_file in pdf_files:
                candidate_id = self.process_cv_file(pdf_file)
                if candidate_id:
                    candidate_ids.append(candidate_id)
            
            logger.info(f"Successfully processed {len(candidate_ids)} CVs")
        
        except Exception as e:
            logger.error(f"Error processing CV directory {cv_dir}: {str(e)}")
            logger.error(traceback.format_exc())
        
        return candidate_ids
    
    def get_candidate_data(self, candidate_id):
        """Get data for a specific candidate.
        
        Args:
            candidate_id (int): Candidate ID in the database
            
        Returns:
            dict: Candidate data
        """
        return self.db_manager.get_candidate(candidate_id)
    
    def get_candidate_by_cv_id(self, cv_id):
        """Get data for a candidate by CV ID.
        
        Args:
            cv_id (str): CV ID (e.g., 'C3014')
            
        Returns:
            dict: Candidate data
        """
        return self.db_manager.get_candidate_by_cv_id(cv_id)
    
    def get_candidate_summary(self, candidate_id):
        """Get a summary of candidate data.
        
        Args:
            candidate_id (int): Candidate ID in the database
            
        Returns:
            dict: Candidate summary
        """
        candidate_data = self.db_manager.get_candidate(candidate_id)
        
        if not candidate_data:
            logger.warning(f"Candidate ID {candidate_id} not found")
            return None
        
        # Create a summarized version
        summary = {
            'id': candidate_data['id'],
            'name': candidate_data['name'],
            'email': candidate_data['email'],
            'skills': candidate_data['skills'],
            'qualifications': candidate_data['qualifications'],
            'experience': [exp[:100] + '...' if len(exp) > 100 else exp 
                          for exp in candidate_data['experience']]
        }
        
        return summary 

# The code below will run when this file is executed directly
if __name__ == "__main__":
    # Configure basic logging to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    print("\n" + "=" * 50)
    print("CV PROCESSOR AGENT - DETAILED OUTPUT DEMO")
    print("=" * 50)
    
    # Create a CV processor instance
    with CVProcessorAgent() as processor:
        # Get the CV directory path
        cv_dir = config.CV_DIR
        print(f"\n[CV PROCESSOR] Looking for CVs in: {cv_dir}")
        
        # Check if the CV directory exists
        if not os.path.isdir(cv_dir):
            print(f"[CV PROCESSOR] Directory does not exist: {cv_dir}")
            print("[CV PROCESSOR] Please check your configuration settings.")
            print(f"[CV PROCESSOR] Current CV_DIR setting: {cv_dir}")
            
            # Try to suggest alternative locations
            parent_dir = Path(__file__).resolve().parent.parent
            dataset_dir = parent_dir / "Dataset"
            print(f"\n[CV PROCESSOR] Trying to locate CV files in: {dataset_dir}")
            
            # Look for a directory that might contain CVs
            if os.path.isdir(dataset_dir):
                possible_cv_dirs = []
                for root, dirs, files in os.walk(dataset_dir):
                    for dir_name in dirs:
                        if "cv" in dir_name.lower() or "resume" in dir_name.lower():
                            possible_cv_dirs.append(os.path.join(root, dir_name))
                
                if possible_cv_dirs:
                    print("\n[CV PROCESSOR] Possible CV directories found:")
                    for dir_path in possible_cv_dirs:
                        print(f"- {dir_path}")
                else:
                    print("[CV PROCESSOR] No CV directories found in the Dataset folder.")
            else:
                print("[CV PROCESSOR] Dataset directory not found.")
        else:
            # Process CVs in the directory
            print("\n[CV PROCESSOR] Processing CVs in directory...")
            
            # List all files in the directory before processing
            all_files = os.listdir(cv_dir)
            pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
            print(f"[CV PROCESSOR] Found {len(pdf_files)} PDF files in directory:")
            for pdf_file in pdf_files:
                print(f"- {pdf_file}")
            
            # Check existing candidates in the database
            try:
                import sqlite3
                db_path = config.DB_PATH
                print(f"\n[CV PROCESSOR] Checking existing candidates in database: {db_path}")
                
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM candidates")
                candidate_count = cursor.fetchone()[0]
                
                print(f"[CV PROCESSOR] Database already contains {candidate_count} candidates")
                
                # Show some sample candidates
                if candidate_count > 0:
                    cursor.execute("SELECT id, cv_id, name FROM candidates ORDER BY id DESC LIMIT 5")
                    recent_candidates = cursor.fetchall()
                    print("[CV PROCESSOR] Most recent candidates in database:")
                    for cand in recent_candidates:
                        print(f"- ID: {cand['id']}, CV_ID: {cand['cv_id']}, Name: {cand['name']}")
                
                conn.close()
            except Exception as e:
                print(f"[CV PROCESSOR] Error checking database: {str(e)}")
            
            # Process CVs
            candidate_ids = processor.process_cv_directory(cv_dir)
            
            if candidate_ids:
                print(f"\n[CV PROCESSOR] Successfully processed {len(candidate_ids)} CVs")
                
                # Check generated emails
                generated_emails_dir = os.path.join(parent_dir, "generated_emails")
                if os.path.isdir(generated_emails_dir):
                    email_files = os.listdir(generated_emails_dir)
                    print(f"\n[CV PROCESSOR] Number of generated email files: {len(email_files)}")
                    
                    # Get timestamps of the latest 5 email files
                    if email_files:
                        email_files.sort(key=lambda x: os.path.getmtime(os.path.join(generated_emails_dir, x)), reverse=True)
                        print("[CV PROCESSOR] Most recent generated emails:")
                        for i, email_file in enumerate(email_files[:5]):
                            print(f"- {email_file}")
                        
                        if len(email_files) > 5:
                            print(f"... and {len(email_files) - 5} more")
                
                # Show a sample candidate
                print("\n[CV PROCESSOR] Sample candidate summary:")
                sample_id = candidate_ids[0]
                summary = processor.get_candidate_summary(sample_id)
                
                if summary:
                    print(f"ID: {summary['id']}")
                    print(f"Name: {summary['name']}")
                    print(f"Email: {summary['email']}")
                    print("Skills:", ", ".join(summary['skills'][:5]) + (", ..." if len(summary['skills']) > 5 else ""))
                    print("Qualifications:", ", ".join(summary['qualifications'][:2]) + (", ..." if len(summary['qualifications']) > 2 else ""))
                    print("Experience Sample:", summary['experience'][0] if summary['experience'] else "None")
                else:
                    print("[CV PROCESSOR] No candidate data available.")
            else:
                print("\n[CV PROCESSOR] No CVs were processed successfully.")
                
                # List files in the directory to help with debugging
                try:
                    all_files = os.listdir(cv_dir)
                    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
                    
                    print(f"\n[CV PROCESSOR] Files in directory: {len(all_files)} total, {len(pdf_files)} PDF files")
                    if pdf_files:
                        print("[CV PROCESSOR] PDF files found:")
                        for pdf in pdf_files[:5]:
                            print(f"- {pdf}")
                        if len(pdf_files) > 5:
                            print(f"... and {len(pdf_files) - 5} more")
                    else:
                        print("[CV PROCESSOR] No PDF files found in the directory.")
                except Exception as e:
                    print(f"[CV PROCESSOR] Error listing directory contents: {str(e)}")
    
    print("\n" + "=" * 50)
    print("CV PROCESSOR AGENT DEMO COMPLETE")
    print("=" * 50) 