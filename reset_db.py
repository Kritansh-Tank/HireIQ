"""
Reset database and re-import job descriptions.

This script will reset the database, clearing all tables, and then
re-import only the job descriptions to get IDs starting from 1.
"""

import os
import sys
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("reset_db")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
from database.db_manager import DBManager
from agents.jd_summarizer import JDSummarizerAgent
import config

def reset_database():
    """Reset the database, clearing all tables."""
    print("\n=== DATABASE RESET ===")
    
    # Create DBManager instance
    db_manager = DBManager()
    
    try:
        # Reset the database - pass True to confirm
        print("Resetting database...")
        success = db_manager.reset_database(confirm=True)
        
        if success:
            # Reset the SQLite sequence counters to start from 1
            print("Resetting SQLite sequence counters...")
            db_manager.db.cursor.execute("DELETE FROM sqlite_sequence")
            db_manager.db.conn.commit()
            print("SQLite sequence counters reset successfully.")
            
            print("Database reset successfully. All data has been cleared.")
        else:
            print("Failed to reset database.")
            return False
            
    except Exception as e:
        print(f"Error resetting database: {str(e)}")
        return False
    finally:
        db_manager.close()
    
    return True

def clean_generated_emails():
    """Delete all generated email files."""
    print("\n=== CLEANING GENERATED EMAILS ===")
    
    # Get path to generated emails directory
    emails_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_emails")
    
    if not os.path.exists(emails_dir):
        print(f"Generated emails directory not found: {emails_dir}")
        return True
    
    try:
        # Count files before deletion
        email_files = [f for f in os.listdir(emails_dir) if os.path.isfile(os.path.join(emails_dir, f))]
        print(f"Found {len(email_files)} email files in {emails_dir}")
        
        # Option 1: Delete all files in the directory
        for file in email_files:
            file_path = os.path.join(emails_dir, file)
            os.remove(file_path)
        
        # Option 2: Remove and recreate the directory
        # shutil.rmtree(emails_dir)
        # os.makedirs(emails_dir, exist_ok=True)
        
        print(f"Successfully deleted {len(email_files)} email files.")
        
    except Exception as e:
        print(f"Error cleaning generated emails: {str(e)}")
        return False
    
    return True

def reimport_job_descriptions():
    """Re-import job descriptions from CSV file."""
    print("\n=== RE-IMPORTING JOB DESCRIPTIONS ===")
    
    # Create JDSummarizerAgent instance
    jd_agent = JDSummarizerAgent()
    
    try:
        # Get the CSV path
        csv_path = config.JD_CSV_PATH
        print(f"Importing job descriptions from: {csv_path}")
        
        # Process job descriptions from CSV
        job_ids = jd_agent.process_job_descriptions_from_csv(csv_path)
        
        if job_ids:
            print(f"Successfully imported {len(job_ids)} job descriptions.")
            print(f"Job IDs range: {min(job_ids)} to {max(job_ids)}")
        else:
            print("No job descriptions were imported.")
            return False
            
    except Exception as e:
        print(f"Error importing job descriptions: {str(e)}")
        return False
    finally:
        jd_agent.close()
    
    return True

def verify_database():
    """Verify database contents after reset and import."""
    print("\n=== VERIFYING DATABASE ===")
    
    # Create DBManager instance
    db_manager = DBManager()
    
    try:
        # Check job descriptions
        all_jobs = db_manager.get_all_jobs()
        print(f"Total jobs in database: {len(all_jobs)}")
        
        if all_jobs:
            print(f"Job IDs range: {all_jobs[0]['id']} to {all_jobs[-1]['id']}")
            print("\nJob Titles:")
            for job in all_jobs:
                print(f"  ID: {job['id']:<3} | Title: {job['title']}")
        
        # Check candidates (should be empty)
        candidates_count = db_manager.db.cursor.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
        print(f"\nTotal candidates in database: {candidates_count}")
        
        # Check match results (should be empty)
        match_count = db_manager.db.cursor.execute('SELECT COUNT(*) FROM match_results').fetchone()[0]
        print(f"Total match results in database: {match_count}")
        
        # Check interview requests (should be empty)
        interview_count = db_manager.db.cursor.execute('SELECT COUNT(*) FROM interview_requests').fetchone()[0]
        print(f"Total interview requests in database: {interview_count}")
        
        # Check generated emails directory
        emails_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_emails")
        if os.path.exists(emails_dir):
            email_files = [f for f in os.listdir(emails_dir) if os.path.isfile(os.path.join(emails_dir, f))]
            print(f"\nGenerated emails directory: {emails_dir}")
            print(f"Email files count: {len(email_files)}")
        
    except Exception as e:
        print(f"Error verifying database: {str(e)}")
        return False
    finally:
        db_manager.close()
    
    return True

def main():
    """Main function."""
    print("\n=== DATABASE RESET AND JOB DESCRIPTIONS REIMPORT ===")
    print("This script will:")
    print("1. Reset the database (delete all data)")
    print("2. Delete all generated email files")
    print("3. Re-import job descriptions to get IDs starting from 1")
    print("\nWARNING: All existing data (candidates, matches, interviews, emails) will be deleted.")
    
    # Ask for confirmation
    confirm = input("\nAre you sure you want to proceed? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Operation cancelled.")
        return
    
    # Reset database
    if not reset_database():
        print("Database reset failed. Aborting.")
        return
    
    # Clean generated emails
    if not clean_generated_emails():
        print("Warning: Failed to clean generated emails.")
    
    # Re-import job descriptions
    if not reimport_job_descriptions():
        print("Job descriptions import failed.")
        return
    
    # Verify database
    verify_database()
    
    print("\n=== OPERATION COMPLETED SUCCESSFULLY ===")
    print("The database has been reset and job descriptions re-imported.")
    print("Job IDs now start from 1.")
    print("All generated email files have been deleted.")
    print("\nYou can now process CVs again using:")
    print("  python app.py --step cv")

if __name__ == "__main__":
    main() 