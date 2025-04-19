"""
Start the AI-Powered Job Application Screening System.
This script starts the frontend web application by default.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/start_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("start_app")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)


def print_header():
    """Print the application header."""
    print("\n" + "=" * 60)
    print("     AI-POWERED JOB APPLICATION SCREENING SYSTEM")
    print("=" * 60)


def run_frontend_app():
    """Run the frontend web application using run_web_app.py."""
    print("\nStarting the frontend web application...")

    try:
        # Run the web app in the current process
        from run_web_app import main as web_app_main
        web_app_main()
        return True
    except Exception as e:
        logger.error(
            f"Error starting frontend web application: {str(e)}", exc_info=True)
        print(f"\nError starting frontend web application: {str(e)}")
        return False


def main():
    """Main function to start the frontend application by default."""
    print_header()
    
    # Simply run the frontend web application
    print("\nStarting the HireIQ web application...")
    run_frontend_app()


if __name__ == "__main__":
    main()
