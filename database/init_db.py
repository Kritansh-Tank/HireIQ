"""
Database initialization script for the AI-Powered Job Application Screening System.
Ensures the database exists with clean tables and loads data from configured sources.
"""

import os
import sys
import logging
import csv
import json
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from database.db_manager import DBManager
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("db_init")

def init_db(reset=True):
    """Initialize the database tables and load data from configured sources.
    
    Args:
        reset (bool): Whether to reset the existing database. 
                     Defaults to True for a clean run each time.
    """
    try:
        logger.info("Initializing database...")
        
        # Create DB manager which will create tables if they don't exist
        db_manager = DBManager()
        
        # Reset the database by default for a clean run
        if reset:
            logger.info("Performing database reset for a clean run")
            db_manager.db.reset_database()
            logger.info("Database reset successfully")
        else:
            logger.warning("Database reset skipped - this will use existing data if available")
        
        # Check if the database path exists
        if os.path.exists(config.DB_PATH):
            db_size = os.path.getsize(config.DB_PATH)
            logger.info(f"Database initialized at {config.DB_PATH} (Size: {db_size/1024:.2f} KB)")
        else:
            logger.warning(f"Database file not found at expected path: {config.DB_PATH}")
        
        # Import data from configured sources
        import_job_descriptions(db_manager)
        import_candidate_cvs(db_manager)
        
        # Check imported data counts
        job_count = len(db_manager.db.get_all_job_descriptions())
        candidate_count = db_manager.db.cursor.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        
        logger.info(f"Database now contains {job_count} job descriptions and {candidate_count} candidates")
        logger.info("Database initialization complete!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)
        return False
    finally:
        if 'db_manager' in locals():
            db_manager.close()

def import_job_descriptions(db_manager):
    """Import job descriptions from the CSV file specified in config."""
    try:
        csv_path = config.JD_CSV_PATH
        if not os.path.exists(csv_path):
            logger.error(f"Job descriptions CSV file not found at: {csv_path}")
            return 0
        
        logger.info(f"Importing job descriptions from {csv_path}")
        count = 0
        
        # Try different encodings - common ones for Windows-created CSV files
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                with open(csv_path, 'r', encoding=encoding) as csvfile:
                    reader = csv.DictReader(csvfile)
                    rows = list(reader)  # Read all rows with this encoding
                    
                    logger.info(f"Successfully read CSV with {encoding} encoding. Found {len(rows)} rows.")
                    
                    for row in rows:
                        # Extract data from CSV row
                        title = row.get('job_title', '')
                        description = row.get('job_description', '')
                        
                        # Parse skills, qualifications and responsibilities
                        # These might be in different formats depending on your CSV
                        skills = row.get('skills', '').split(',') if 'skills' in row else []
                        qualifications = row.get('qualifications', '').split(',') if 'qualifications' in row else []
                        responsibilities = row.get('responsibilities', '').split(',') if 'responsibilities' in row else []
                        
                        # Clean up the lists
                        skills = [s.strip() for s in skills if s.strip()]
                        qualifications = [q.strip() for q in qualifications if q.strip()]
                        responsibilities = [r.strip() for r in responsibilities if r.strip()]
                        
                        # Import the job description
                        if title and description:
                            job_id = db_manager.import_job_description(
                                title=title,
                                description=description,
                                skills=skills,
                                qualifications=qualifications,
                                responsibilities=responsibilities
                            )
                            if job_id:
                                count += 1
                    
                    # If we get here, the encoding worked
                    logger.info(f"Successfully imported {count} job descriptions using {encoding} encoding")
                    return count
                
            except UnicodeDecodeError as e:
                logger.warning(f"Failed to read CSV with {encoding} encoding: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Error processing CSV with {encoding} encoding: {str(e)}")
                continue
        
        # If we get here, none of the encodings worked
        logger.error("Failed to read CSV with any of the attempted encodings")
        return 0
        
    except Exception as e:
        logger.error(f"Error importing job descriptions: {str(e)}", exc_info=True)
        return 0

def import_candidate_cvs(db_manager):
    """Import candidate CVs from the directory specified in config."""
    try:
        cv_dir = config.CV_DIR
        if not os.path.exists(cv_dir):
            logger.error(f"CV directory not found at: {cv_dir}")
            return 0
        
        logger.info(f"Importing candidate CVs from {cv_dir}")
        count = 0
        
        # Get a list of CV files (PDF format)
        cv_files = [f for f in os.listdir(cv_dir) if os.path.isfile(os.path.join(cv_dir, f)) and f.lower().endswith('.pdf')]
        
        if not cv_files:
            logger.warning(f"No PDF files found in {cv_dir}")
            return 0
            
        # Check if PyPDF2 is installed
        try:
            import PyPDF2
        except ImportError:
            logger.error("PyPDF2 package is required for reading PDF files. Please install using 'pip install PyPDF2'")
            return 0
        
        for cv_file in cv_files:
            try:
                # Extract CV ID from filename
                cv_id = os.path.splitext(cv_file)[0]
                
                # Read PDF content
                cv_path = os.path.join(cv_dir, cv_file)
                cv_text = ""
                
                try:
                    with open(cv_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            cv_text += page.extract_text() + "\n"
                    
                    if not cv_text.strip():
                        logger.warning(f"No text could be extracted from PDF file {cv_file} - it may be scanned or image-based")
                        continue
                        
                    logger.debug(f"Successfully extracted {len(cv_text)} characters from PDF file {cv_file}")
                    
                except Exception as e:
                    logger.error(f"Error reading PDF file {cv_file}: {str(e)}")
                    continue
                
                # Process the CV to extract data
                # This is a simplified version - you might need to adjust based on your actual CV format
                name = cv_id  # Default name is the CV ID
                email = ""
                phone = ""
                
                # Extract name if available (this is just a sample approach)
                name_line = cv_text.split('\n')[0] if cv_text else cv_id
                if name_line and len(name_line.strip()) > 0:
                    name = name_line.strip()
                
                # For demo purposes, we'll just use some placeholder data
                # In a real system, you would extract this from the CV text
                skills = ["Python", "Java", "SQL", "Communication"]
                qualifications = ["Bachelor's Degree", "Certifications"]
                experience = [
                    {"title": "Software Developer", "company": "Tech Company", "years": 2}
                ]
                
                # Import the candidate
                candidate_id = db_manager.import_candidate(
                    cv_id=cv_id,
                    name=name,
                    email=email,
                    phone=phone,
                    skills=skills,
                    qualifications=qualifications,
                    experience=experience,
                    cv_text=cv_text
                )
                
                if candidate_id:
                    count += 1
            except Exception as e:
                logger.error(f"Error processing CV file {cv_file}: {str(e)}")
                continue
        
        logger.info(f"Successfully imported {count} candidates")
        return count
    except Exception as e:
        logger.error(f"Error importing candidate CVs: {str(e)}", exc_info=True)
        return 0

if __name__ == "__main__":
    init_db(reset=True)
    print("Database initialization complete with reset and data import. You can now start the application.") 