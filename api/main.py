"""
FastAPI application for the AI-Powered Job Application Screening System.
"""

import os
import time
import logging
import threading
import asyncio
from datetime import datetime
from fastapi import FastAPI, Request, Depends, HTTPException, Form, UploadFile, File, Body
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pathlib import Path
import sys
import json
import uvicorn
from typing import List, Dict, Optional, Any

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from database.db_manager import DBManager
from efficient_matcher import EfficientMatcher
from utils.email_sender import (
    get_all_emails_data, 
    send_emails_to_candidates, 
    save_email_configuration,
    load_email_configuration
)
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, "logs", "api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api")

# Process status
process_status = {
    "status": "idle",  # idle, running, completed
    "progress": 0,
    "message": "",
    "results": None,
    "started_at": None,
    "completed_at": None,
    "job_ids": []  # Store job IDs from the backend process
}

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Job Application Screening System",
    description="A system that uses AI to process CVs, match candidates with job descriptions, and streamline the hiring process.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip middleware for compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
static_dir = Path(__file__).resolve().parent / "static"
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize templates
templates_dir = Path(__file__).resolve().parent / "templates"
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Database dependency
def get_db():
    db = DBManager()
    try:
        yield db
    finally:
        db.close()

# Initialize EfficientMatcher
def get_matcher():
    matcher = EfficientMatcher()
    try:
        yield matcher
    finally:
        matcher.close()

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request to {request.url.path} processed in {process_time:.4f} seconds")
    return response

# Custom exception handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    if request.headers.get("accept") == "application/json":
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "message": exc.detail}
        )
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {str(exc)}")
    if request.headers.get("accept") == "application/json":
        return JSONResponse(
            status_code=422,
            content={
                "status": "error", 
                "message": "Validation error", 
                "details": exc.errors()
            }
        )
    return await request_validation_exception_handler(request, exc)

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    if request.headers.get("accept") == "application/json":
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "An unexpected error occurred"}
        )
    return templates.TemplateResponse(
        "error.html", 
        {"request": request, "status_code": 500, "detail": "An unexpected error occurred"}
    )

# Main background process function
def run_screening_process():
    """Run the entire job screening process."""
    global process_status
    
    try:
        logger.info("Starting job screening process using the backend pipeline")
        process_status["message"] = "Initializing the backend pipeline..."
        
        # Perform a thorough database reset before starting
        process_status["message"] = "Resetting database..."
        process_status["progress"] = 2
        
        try:
            # Direct database reset without importing reset_db.py
            logger.info("Directly resetting database tables...")
            
            # Create a new database connection
            with DBManager() as db_manager:
                # Reset the database - pass True to confirm
                reset_result = db_manager.reset_database(confirm=True)
                
                if not reset_result:
                    logger.error("Failed to reset database tables")
                    process_status["status"] = "idle"
                    process_status["progress"] = 100
                    process_status["message"] = "Process failed - Database reset error"
                    process_status["results"] = "<div class='alert alert-danger'><h4>Database Reset Failed</h4><p>Could not reset the database before starting the process.</p></div>"
                    process_status["completed_at"] = datetime.now()
                    return
                    
                # Reset the SQLite sequence counters to start from 1
                logger.info("Resetting SQLite sequence counters...")
                db_manager.db.cursor.execute("DELETE FROM sqlite_sequence")
                db_manager.db.conn.commit()
            
            # Clean generated emails directory
            try:
                logger.info("Cleaning generated emails directory...")
                emails_dir = os.path.join(parent_dir, "generated_emails")
                
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
            
            logger.info("Database reset successful, all previous data cleared")
        except Exception as reset_err:
            logger.error(f"Error during database reset: {str(reset_err)}", exc_info=True)
            # Continue anyway despite the reset error
            logger.warning("Continuing process despite database reset error")
        
        # Import the OptimizedJobScreeningApp from app.py
        from app import OptimizedJobScreeningApp
        
        # Update status
        process_status["progress"] = 5
        process_status["message"] = "Backend pipeline initialized, starting processing..."
        
        # Set up the optimized app with good performance defaults
        with OptimizedJobScreeningApp(
            use_parallel=True,
            max_workers=8,
            batch_size=20,
            use_llm_for_top_k=5
        ) as app:
            # Step 1: Process job descriptions
            process_status["progress"] = 15
            process_status["message"] = "Processing job descriptions..."
            job_ids = app.process_job_descriptions()
            
            if not job_ids:
                logger.warning("No job descriptions found or imported")
                process_status["status"] = "idle"
                process_status["progress"] = 100
                process_status["message"] = "Process completed - No job descriptions found"
                process_status["results"] = "<div class='alert alert-warning'><h4>No Job Descriptions Found</h4><p>Please check that your job descriptions CSV file is correctly configured and accessible.</p></div>"
                process_status["completed_at"] = datetime.now()
                return
            
            # Step 2: Process CVs
            process_status["progress"] = 30
            process_status["message"] = f"Successfully processed {len(job_ids)} job descriptions. Now processing candidate CVs..."
            candidate_ids = app.process_cvs()
            
            if not candidate_ids:
                logger.warning("No candidate CVs found or imported")
                process_status["status"] = "idle"
                process_status["progress"] = 100
                process_status["message"] = "Process completed - No candidate CVs found"
                process_status["results"] = "<div class='alert alert-warning'><h4>No Candidate CVs Found</h4><p>Please check that your CV directory is correctly configured and accessible.</p></div>"
                process_status["completed_at"] = datetime.now()
                return
                
            # Step 3: Precompute embeddings
            process_status["progress"] = 45
            process_status["message"] = f"Successfully processed {len(candidate_ids)} candidate CVs. Now precomputing embeddings..."
            app.precompute_embeddings(job_ids, candidate_ids)
            
            # Step 4: Match jobs with candidates
            process_status["progress"] = 60
            process_status["message"] = "Matching jobs with candidates..."
            match_results = app.match_jobs_with_candidates(
                job_ids=job_ids,
                candidate_ids=candidate_ids,
                enhance_top_k=5
            )
            
            # Step 5: Shortlist candidates
            process_status["progress"] = 80
            process_status["message"] = "Shortlisting candidates for each job..."
            shortlist_counts = app.shortlist_candidates(job_ids)
            
            # Step 6: Schedule interviews
            process_status["progress"] = 90
            process_status["message"] = "Scheduling interviews for shortlisted candidates..."
            interview_counts = app.schedule_interviews(job_ids)
            
            # Complete the process
            process_status["progress"] = 100
            
            # Create a simple HTML summary
            total_jobs = len(job_ids)
            total_candidates = len(candidate_ids)
            total_shortlisted = sum(shortlist_counts.values())
            
            if isinstance(interview_counts, dict) and 'total' in interview_counts:
                total_interviews = interview_counts['total']
            else:
                total_interviews = sum(interview_counts.values())
                
            results_html = f"""
            <div class="alert alert-success">
                <h4><i class="fas fa-check-circle me-2"></i> Process Completed Successfully</h4>
                <p>The job screening process has been completed successfully.</p>
            </div>
            <div class="row">
                <div class="col-md-3">
                    <div class="card text-center mb-3">
                        <div class="card-body">
                            <h1 class="display-4">{total_jobs}</h1>
                            <p class="text-muted">Jobs Processed</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-center mb-3">
                        <div class="card-body">
                            <h1 class="display-4">{total_candidates}</h1>
                            <p class="text-muted">Candidates Processed</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-center mb-3">
                        <div class="card-body">
                            <h1 class="display-4">{total_shortlisted}</h1>
                            <p class="text-muted">Candidates Shortlisted</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-center mb-3">
                        <div class="card-body">
                            <h1 class="display-4">{total_interviews}</h1>
                            <p class="text-muted">Interviews Scheduled</p>
                        </div>
                    </div>
                </div>
            </div>
            """
            
            # Update the final status
            process_status["status"] = "idle"
            process_status["message"] = "Process completed successfully"
            process_status["results"] = results_html
            process_status["completed_at"] = datetime.now()
            
            process_status["job_ids"] = job_ids
            
            logger.info("Screening process completed successfully")
            
    except Exception as e:
        logger.error(f"Error in screening process: {str(e)}", exc_info=True)
        process_status["status"] = "idle"
        process_status["progress"] = 100
        process_status["message"] = f"Error: {str(e)}"
        process_status["results"] = f"""
        <div class="alert alert-danger">
            <h4><i class="fas fa-exclamation-triangle me-2"></i> Error</h4>
            <p>An error occurred during the screening process:</p>
            <pre>{str(e)}</pre>
        </div>
        """
        process_status["completed_at"] = datetime.now()

def generate_summary_html(job_matches):
    """Generate HTML summary of job matching results"""
    
    if not job_matches:
        return "<div class='alert alert-warning'><h4>No Matching Results Available</h4><p>The screening process did not produce any matches. This could be due to:</p><ul><li>No job descriptions in the database</li><li>No candidate CVs in the database</li><li>Issues with the matching algorithm</li></ul><p>Please check your data sources and configuration.</p></div>"
    
    # Count matches
    total_jobs = len(job_matches)
    
    # Handle case where jobs have no candidate matches
    empty_jobs = [job_id for job_id, data in job_matches.items() if not data.get("candidates")]
    if len(empty_jobs) == total_jobs:
        return "<div class='alert alert-warning'><h4>No Candidates Matched</h4><p>We found job descriptions but no candidates matched with them. This could be due to:</p><ul><li>No candidate CVs in the database</li><li>Very strict matching criteria</li><li>Issues with the matching algorithm</li></ul><p>Please check your data sources and configuration.</p></div>"
    
    # Calculate totals for the summary cards
    total_matches = sum(len(data["candidates"]) for data in job_matches.values())
    
    # Use a set to avoid counting duplicate candidates
    unique_candidates = set()
    for data in job_matches.values():
        for candidate_match in data["candidates"]:
            unique_candidates.add(candidate_match["candidate"]["id"])
    total_candidates = len(unique_candidates)
    
    html = f"""
    <div class="mb-4">
        <div class="row mb-3">
            <div class="col-md-4">
                <div class="card bg-primary text-white">
                    <div class="card-body text-center py-4">
                        <h3 class="display-4">{total_jobs}</h3>
                        <p class="lead">Jobs Processed</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-success text-white">
                    <div class="card-body text-center py-4">
                        <h3 class="display-4">{total_candidates}</h3>
                        <p class="lead">Candidates Processed</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-info text-white">
                    <div class="card-body text-center py-4">
                        <h3 class="display-4">{total_matches}</h3>
                        <p class="lead">Total Matches</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    # If there are no matches to display, add an explanation
    if total_matches == 0:
        html += """
        <div class="alert alert-info">
            <h4>No Matching Candidates Found</h4>
            <p>We processed the jobs and candidates, but no matches were found that meet the criteria.</p>
            <p>You may want to check:</p>
            <ul>
                <li>The matching threshold might be too high</li>
                <li>The candidate skills might not align with job requirements</li>
                <li>There might be issues with how the data is being processed</li>
            </ul>
        </div>
        """
        return html
    
    # Add job-specific results
    html += '<div class="accordion" id="jobAccordion">'
    
    for i, (job_id, data) in enumerate(job_matches.items()):
        job = data["job"]
        candidates = data["candidates"]
        
        # Skip jobs with no candidates
        if not candidates:
            continue
        
        # Calculate statistics
        high_matches = len([c for c in candidates if c["match"]["match_score"] >= 0.8])
        medium_matches = len([c for c in candidates if 0.6 <= c["match"]["match_score"] < 0.8])
        low_matches = len([c for c in candidates if c["match"]["match_score"] < 0.6])
        
        html += f"""
        <div class="accordion-item">
            <h2 class="accordion-header" id="heading{job_id}">
                <button class="accordion-button {'collapsed' if i > 0 else ''}" type="button" data-bs-toggle="collapse" 
                        data-bs-target="#collapse{job_id}" aria-expanded="{str(i == 0).lower()}" aria-controls="collapse{job_id}">
                    <strong>{job['title']}</strong> ({len(candidates)} candidate matches)
                </button>
            </h2>
            <div id="collapse{job_id}" class="accordion-collapse collapse {'show' if i == 0 else ''}" 
                 aria-labelledby="heading{job_id}" data-bs-parent="#jobAccordion">
                <div class="accordion-body">
                    <div class="mb-3">
                        <h5>Match Quality Distribution</h5>
                        <div class="row">
                            <div class="col-md-4">
                                <div class="d-flex align-items-center mb-2">
                                    <div style="width: 100px;">High (80%+)</div>
                                    <div class="progress flex-grow-1">
                                        <div class="progress-bar match-high" role="progressbar" 
                                            style="width: {high_matches/max(1, len(candidates))*100}%" 
                                            aria-valuenow="{high_matches}" aria-valuemin="0" 
                                            aria-valuemax="{len(candidates)}">
                                            {high_matches}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="d-flex align-items-center mb-2">
                                    <div style="width: 100px;">Med (60-79%)</div>
                                    <div class="progress flex-grow-1">
                                        <div class="progress-bar match-medium" role="progressbar" 
                                            style="width: {medium_matches/max(1, len(candidates))*100}%" 
                                            aria-valuenow="{medium_matches}" aria-valuemin="0" 
                                            aria-valuemax="{len(candidates)}">
                                            {medium_matches}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="d-flex align-items-center mb-2">
                                    <div style="width: 100px;">Low (<60%)</div>
                                    <div class="progress flex-grow-1">
                                        <div class="progress-bar match-low" role="progressbar" 
                                            style="width: {low_matches/max(1, len(candidates))*100}%" 
                                            aria-valuenow="{low_matches}" aria-valuemin="0" 
                                            aria-valuemax="{len(candidates)}">
                                            {low_matches}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h5>Top Candidates</h5>
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Candidate</th>
                                    <th>Match Score</th>
                                    <th>Skills Match</th>
                                    <th>Experience Match</th>
                                </tr>
                            </thead>
                            <tbody>
        """
        
        # Add top 5 candidates (or all if less than 5)
        top_candidates = candidates[:min(5, len(candidates))]
        for idx, candidate_data in enumerate(top_candidates):
            candidate = candidate_data["candidate"]
            match = candidate_data["match"]
            
            html += f"""
                <tr>
                    <td>{idx + 1}</td>
                    <td>{candidate.get('name', 'Candidate #' + str(candidate['id']))}</td>
                    <td>
                        <div class="d-flex align-items-center">
                            <div class="me-2">{match['match_score']*100:.1f}%</div>
                            <div class="progress flex-grow-1">
                                <div class="progress-bar {'match-high' if match['match_score'] >= 0.8 else 'match-medium' if match['match_score'] >= 0.6 else 'match-low'}" 
                                     role="progressbar" style="width: {match['match_score']*100}%"></div>
                            </div>
                        </div>
                    </td>
                    <td>{match['skills_match_score']*100:.1f}%</td>
                    <td>{match['experience_match_score']*100:.1f}%</td>
                </tr>
            """
        
        # Close table and accordion for this job
        html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        """
    
    # Close accordion
    html += '</div>'
    
    return html

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with simplified job screening dashboard."""
    try:
        return templates.TemplateResponse(
            "dashboard.html", 
            {
                "request": request, 
                "process_status": process_status["status"],
                "progress": process_status["progress"],
                "status_message": process_status["message"],
                "results": process_status["results"]
            }
        )
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error loading dashboard")

@app.post("/api/start-process")
async def start_process():
    """API endpoint to start the job screening process."""
    global process_status
    
    # Check if already running
    if process_status["status"] == "running":
        logger.warning("Process start requested but process is already running")
        return {"status": "error", "message": "Process is already running"}
    
    try:
        # Note: Database reset is now handled in the run_screening_process function
        
        # Reset status
        process_status = {
            "status": "running",
            "progress": 0,
            "message": "Starting process...",
            "results": None,
            "started_at": datetime.now(),
            "completed_at": None,
            "job_ids": []
        }
        
        logger.info("Starting background process for job screening")
        
        # Start background process
        thread = threading.Thread(target=run_screening_process)
        thread.daemon = True  # Allow the thread to be terminated when the main program exits
        thread.start()
        
        logger.info("Background process thread started successfully")
        
        return {
            "status": "started", 
            "message": "Process started successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to start process: {str(e)}", exc_info=True)
        process_status["status"] = "idle"
        process_status["message"] = f"Failed to start: {str(e)}"
        return {"status": "error", "message": f"Failed to start process: {str(e)}"}

@app.get("/api/process-status")
async def get_process_status():
    """API endpoint to get the current status of the job screening process."""
    return {
        "status": process_status["status"],
        "progress": process_status["progress"],
        "message": process_status["message"],
        "results": process_status["results"],
        "started_at": process_status["started_at"].isoformat() if process_status["started_at"] else None,
        "completed_at": process_status["completed_at"].isoformat() if process_status["completed_at"] else None
    }

@app.get("/job/{job_id}", response_class=HTMLResponse)
async def job_details(
    request: Request, 
    job_id: int, 
    db: DBManager = Depends(get_db)
):
    """Job details page with matched candidates."""
    try:
        job = db.db.get_job_description(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get matches for this job
        matches = db.db.get_match_results(job_id)
        
        # Get candidate details for each match
        candidates_with_matches = []
        for match in matches:
            candidate = db.db.get_candidate(match["candidate_id"])
            if candidate:
                candidates_with_matches.append({
                    "candidate": candidate,
                    "match": match
                })
        
        # Sort by match score (descending)
        candidates_with_matches.sort(key=lambda x: x["match"]["match_score"], reverse=True)
        
        return templates.TemplateResponse(
            "job_details.html", 
            {
                "request": request, 
                "job": job,
                "candidates": candidates_with_matches
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in job_details route: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving job details")

@app.get("/candidate/{candidate_id}", response_class=HTMLResponse)
async def candidate_details(
    request: Request, 
    candidate_id: int, 
    db: DBManager = Depends(get_db)
):
    """Candidate details page."""
    try:
        candidate = db.db.get_candidate(candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        # Get all matches for this candidate
        matches = []
        jobs = db.db.get_all_job_descriptions()
        for job in jobs:
            job_matches = db.db.get_match_results(job["id"])
            for match in job_matches:
                if match["candidate_id"] == candidate_id:
                    matches.append({
                        "job": job,
                        "match": match
                    })
        
        # Sort by match score (descending)
        matches.sort(key=lambda x: x["match"]["match_score"], reverse=True)
        
        return templates.TemplateResponse(
            "candidate_details.html", 
            {
                "request": request, 
                "candidate": candidate,
                "matches": matches
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in candidate_details route: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving candidate details")

@app.get("/api/jobs", response_model=List[dict])
async def api_get_jobs(db: DBManager = Depends(get_db)):
    """API endpoint to get all jobs."""
    try:
        return db.db.get_all_job_descriptions()
    except Exception as e:
        logger.error(f"Error in api_get_jobs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving jobs")

@app.get("/api/job/{job_id}", response_model=dict)
async def api_get_job(job_id: int, db: DBManager = Depends(get_db)):
    """API endpoint to get job details."""
    try:
        job = db.db.get_job_description(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in api_get_job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving job details")

@app.get("/api/job/{job_id}/matches", response_model=List[dict])
async def api_get_job_matches(job_id: int, db: DBManager = Depends(get_db)):
    """API endpoint to get matches for a job."""
    try:
        job = db.db.get_job_description(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get matches for this job
        matches = db.db.get_match_results(job_id)
        
        # Get candidate details for each match
        results = []
        for match in matches:
            candidate = db.db.get_candidate(match["candidate_id"])
            if candidate:
                results.append({
                    "candidate": candidate,
                    "match": match
                })
        
        # Sort by match score (descending)
        results.sort(key=lambda x: x["match"]["match_score"], reverse=True)
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in api_get_job_matches: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving job matches")

@app.get("/api/candidates", response_model=List[dict])
async def api_get_candidates(db: DBManager = Depends(get_db)):
    """API endpoint to get all candidates."""
    try:
        candidates = db.db.cursor.execute("SELECT * FROM candidates").fetchall()
        return [dict(candidate) for candidate in candidates]
    except Exception as e:
        logger.error(f"Error in api_get_candidates: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving candidates")

@app.get("/api/candidate/{candidate_id}", response_model=dict)
async def api_get_candidate(candidate_id: int, db: DBManager = Depends(get_db)):
    """API endpoint to get candidate details."""
    try:
        candidate = db.db.get_candidate(candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        return candidate
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in api_get_candidate: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving candidate details")

@app.post("/api/run-matching")
async def api_run_matching(
    job_id: Optional[int] = None,
    matcher: EfficientMatcher = Depends(get_matcher),
    db: DBManager = Depends(get_db)
):
    """API endpoint to run the matching process for a specific job or all jobs."""
    try:
        if job_id:
            # Check if job exists
            job = db.db.get_job_description(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            
            # Run matching for this job
            logger.info(f"Running matching for job ID {job_id}")
            matches = matcher.match_job_with_all_candidates(job_id)
            logger.info(f"Completed matching for job ID {job_id}, found {len(matches)} matches")
            return {"status": "success", "job_id": job_id, "matches_count": len(matches)}
        else:
            # Run matching for all jobs
            all_jobs = db.db.get_all_job_descriptions()
            job_ids = [job["id"] for job in all_jobs]
            
            logger.info(f"Running matching for all {len(job_ids)} jobs")
            total_matches = 0
            for job_id in job_ids:
                matches = matcher.match_job_with_all_candidates(job_id)
                total_matches += len(matches)
            
            logger.info(f"Completed matching for all jobs, total matches: {total_matches}")
            return {"status": "success", "jobs_count": len(job_ids), "total_matches": total_matches}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in api_run_matching: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/api/send-emails")
async def send_emails(
    job_id: Optional[int] = None,
    process_all: bool = False
):
    """API endpoint to send emails to candidates for a specific job or all jobs."""
    try:
        # Create a new database connection specifically for this request
        # This avoids the SQLite thread safety issue
        from database.db_manager import DBManager
        
        # Use context manager to ensure proper cleanup
        with DBManager() as db:
            job_ids_to_process = []
            
            # If no job_id is provided or process_all is true, use all job IDs from the process results
            if job_id is None or process_all:
                if not process_status["job_ids"]:
                    return {
                        "status": "error", 
                        "message": "No jobs available. Please run the screening process first."
                    }
                job_ids_to_process = process_status["job_ids"]
                logger.info(f"Processing all {len(job_ids_to_process)} jobs")
            else:
                job_ids_to_process = [job_id]
                logger.info(f"Processing single job with ID: {job_id}")
            
            if not job_ids_to_process:
                return {
                    "status": "error", 
                    "message": "No jobs to process."
                }
            
            # Process each job
            job_results = []
            total_sent = 0
            total_failed = 0
            total_candidates = 0
            
            for current_job_id in job_ids_to_process:
                logger.info(f"Processing emails for job ID: {current_job_id}")
                
                job = db.db.get_job_description(current_job_id)
                if not job:
                    logger.error(f"Job with ID {current_job_id} not found")
                    job_results.append({
                        "job_id": current_job_id,
                        "status": "error",
                        "message": f"Job with ID {current_job_id} not found"
                    })
                    continue
                
                logger.info(f"Found job: {job['title']} (ID: {current_job_id})")
                
                # Get the company name from config if available, otherwise use a default
                company_name = getattr(config, 'COMPANY_NAME', 'HireIQ Recruitment Team')
                
                # Get matches for this job
                matches = db.db.get_match_results(current_job_id)
                logger.info(f"Found {len(matches)} matches for job {current_job_id}")
                
                # Get candidate details for each match
                candidates_with_matches = []
                candidates_without_email = 0
                
                for match in matches:
                    candidate = db.db.get_candidate(match["candidate_id"])
                    if candidate:
                        if not candidate.get('email'):
                            candidates_without_email += 1
                            logger.warning(f"Candidate {candidate['id']} ({candidate.get('name', 'Unknown')}) has no email address")
                        else:
                            candidates_with_matches.append({
                                "candidate": candidate,
                                "match": match
                            })
                    else:
                        logger.warning(f"Candidate {match['candidate_id']} not found in database")
                
                logger.info(f"Found {len(candidates_with_matches)} candidates with valid email addresses for job {current_job_id}")
                
                # Sort by match score (descending)
                candidates_with_matches.sort(key=lambda x: x["match"]["match_score"], reverse=True)
                
                sent_count = 0
                failed_count = 0
                
                # Send emails to candidates
                for candidate_data in candidates_with_matches:
                    candidate = candidate_data["candidate"]
                    match = candidate_data["match"]
                    
                    # This check is redundant now but kept for safety
                    if not candidate.get('email'):
                        failed_count += 1
                        continue
                        
                    # Create email subject
                    subject = f"You've been shortlisted for {job['title']}"
                    
                    # Create email content
                    email_content = (
                        f"Dear {candidate['name']},\n\n"
                        f"We are pleased to inform you that you have been shortlisted for the position of {job['title']}.\n\n"
                        f"Here are the details of the job:\n\n"
                        f"- Job Title: {job['title']}\n"
                        f"- Job Description: {job['description']}\n"
                        f"- Skills Required: {', '.join(job['skills'])}\n"
                        f"- Key Responsibilities: {', '.join(job['responsibilities'])}\n\n"
                        f"We believe you are a good fit for this position based on your skills and experience.\n\n"
                        f"Please let us know if you are interested in this opportunity.\n\n"
                        f"Best regards,\n"
                        f"{company_name}\n\n"
                        f"This email is automatically generated, please do not reply to this email."
                    )
                    
                    # Use the email sending utility
                    from utils.email_sender import send_email
                    logger.info(f"Sending email to {candidate['email']} for candidate {candidate['id']} ({candidate['name']})")
                    
                    success = send_email(
                        recipient=candidate['email'],
                        subject=subject,
                        body=email_content
                    )
                    
                    if success:
                        sent_count += 1
                        total_sent += 1
                        logger.info(f"Successfully sent email to {candidate['email']}")
                    else:
                        failed_count += 1
                        total_failed += 1
                        logger.error(f"Failed to send email to {candidate['email']}")
                
                total_candidates += len(candidates_with_matches)
                
                # Determine why no emails were sent for this job
                if len(candidates_with_matches) == 0:
                    if len(matches) == 0:
                        diagnosis = "No candidates matched with this job."
                    else:
                        diagnosis = f"Found {len(matches)} candidate matches, but {candidates_without_email} had no email address."
                else:
                    diagnosis = ""
                
                # Add results for this job
                job_results.append({
                    "job_id": current_job_id,
                    "job_title": job['title'],
                    "status": "success",
                    "total_matches": len(matches),
                    "candidates_with_email": len(candidates_with_matches),
                    "candidates_without_email": candidates_without_email,
                    "emails_sent": sent_count,
                    "emails_failed": failed_count,
                    "diagnosis": diagnosis
                })
            
            # Create overall summary
            if len(job_ids_to_process) == 1:
                job_result = job_results[0]
                message = f"Processed {job_result['candidates_with_email']} emails for job '{job_result['job_title']}'"
                detailed_results = {
                    "job_id": job_result['job_id'],
                    "job_title": job_result['job_title'],
                    "total_matches": job_result['total_matches'],
                    "candidates_with_email": job_result['candidates_with_email'],
                    "candidates_without_email": job_result['candidates_without_email'],
                    "diagnosis": job_result['diagnosis']
                }
            else:
                message = f"Processed {total_candidates} emails for {len(job_ids_to_process)} jobs"
                detailed_results = {
                    "jobs_processed": len(job_ids_to_process),
                    "total_matches": sum(job['total_matches'] for job in job_results),
                    "total_candidates_with_email": sum(job['candidates_with_email'] for job in job_results),
                    "total_candidates_without_email": sum(job['candidates_without_email'] for job in job_results),
                }
            
            return {
                "status": "success", 
                "message": message,
                "sent": total_sent,
                "failed": total_failed,
                "job_results": job_results,
                **detailed_results
            }
    except Exception as e:
        logger.error(f"Error in send_emails: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.get("/api/emails")
async def list_emails():
    """API endpoint to list all emails."""
    try:
        emails = get_all_emails_data()
        return {"status": "success", "emails": emails}
    except Exception as e:
        logger.error(f"Error in list_emails: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.post("/api/save-email-config")
async def save_email_config(config_data: Dict[str, Any] = Body(...)):
    """API endpoint to save email configuration."""
    try:
        success = save_email_configuration(config_data)
        if success:
            return {"status": "success", "message": "Email configuration saved successfully"}
        else:
            return {"status": "error", "message": "Failed to save email configuration"}
    except Exception as e:
        logger.error(f"Error in save_email_config: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.get("/api/email-config")
async def get_email_config():
    """API endpoint to get current email configuration."""
    try:
        config = load_email_configuration()
        # Remove password from response for security
        if 'password' in config:
            config['password'] = '**********' if config['password'] else ''
        return {"status": "success", "config": config}
    except Exception as e:
        logger.error(f"Error in get_email_config: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)