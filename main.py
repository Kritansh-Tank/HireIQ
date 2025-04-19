"""
Main entry point for the AI-Powered Job Application Screening System.
Combines the FastAPI application with the existing job screening functionality.
"""

from api.main import app
from database.init_db import init_db
import sys
import os
import uvicorn
from pathlib import Path
import logging
from database.db_manager import DBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

# Add the current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

def force_reset_database():
    """Force a complete database reset and clear sequence counters."""
    logger.info("Performing forced database reset on application startup...")
    
    try:
        # Create a new database connection
        with DBManager() as db_manager:
            # Reset the database tables
            reset_result = db_manager.reset_database(confirm=True)
            
            if not reset_result:
                logger.error("Failed to reset database tables")
                return False
                
            # Reset the SQLite sequence counters to start from 1
            logger.info("Resetting SQLite sequence counters...")
            db_manager.db.cursor.execute("DELETE FROM sqlite_sequence")
            db_manager.db.conn.commit()
            
            # Verify reset worked
            job_count = db_manager.db.cursor.execute("SELECT COUNT(*) FROM job_descriptions").fetchone()[0]
            candidate_count = db_manager.db.cursor.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            match_count = db_manager.db.cursor.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]
            interview_count = db_manager.db.cursor.execute("SELECT COUNT(*) FROM interview_requests").fetchone()[0]
            
            logger.info(f"Database reset complete. Tables emptied: jobs={job_count}, candidates={candidate_count}, matches={match_count}, interviews={interview_count}")
            
        # Clean generated emails directory
        try:
            logger.info("Cleaning generated emails directory...")
            emails_dir = os.path.join(current_dir, "generated_emails")
            
            if os.path.exists(emails_dir):
                # Count files before deletion
                email_files = [f for f in os.listdir(emails_dir) if os.path.isfile(os.path.join(emails_dir, f))]
                logger.info(f"Found {len(email_files)} email files in {emails_dir}")
                
                # Delete all files in the directory
                for file in email_files:
                    file_path = os.path.join(emails_dir, file)
                    os.remove(file_path)
                
                logger.info(f"Successfully deleted {len(email_files)} email files")
        except Exception as email_err:
            logger.error(f"Error cleaning generated emails: {str(email_err)}")
            # Continue despite email cleaning errors
        
        return True
        
    except Exception as e:
        logger.error(f"Error during forced database reset: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting AI-Powered Job Application Screening System...")

    # Force reset the database at startup
    print("Forcing database reset to ensure clean state...")
    if force_reset_database():
        print("Database reset successfully completed.")
    else:
        print("WARNING: Database reset failed. IDs may not start from 1.")

    # Initialize the database with job descriptions
    print("Initializing database...")
    init_db(reset=False)  # Don't reset again, just initialize

    print("Database initialized. Starting API server...")
    print("Once started, access the application at: http://localhost:8000")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
