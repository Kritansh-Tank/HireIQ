"""
Run the web application for the AI-Powered Job Application Screening System.
This script starts the FastAPI server that serves the web UI and API endpoints.
"""

import os
import sys
import logging
import uvicorn
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/web_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("web_app")

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)


def main():
    """Start the FastAPI web application."""
    logger.info("Starting web application...")

    # Create required directories
    api_dir = Path("api")
    templates_dir = api_dir / "templates"
    static_dir = api_dir / "static"

    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    # Check if config file exists
    if not os.path.exists("config.py"):
        logger.error(
            "config.py not found. Please ensure the configuration file exists.")
        return

    # Check if the API module exists
    if not os.path.exists(api_dir / "main.py"):
        logger.error("API module not found. Please ensure api/main.py exists.")
        return

    # Start the FastAPI server
    try:
        logger.info("Starting FastAPI server...")

        # Print a welcome message
        print("\n===== AI-Powered Job Application Screening System =====")
        print("Frontend UI will be available at: http://localhost:8000")
        print("==============================================================\n")

        # Start the server using uvicorn
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    except Exception as e:
        logger.error(
            f"Error starting web application: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
