"""
AI-Powered Job Application Screening System (Optimized Version)

This is the optimized version of the application entry point that runs the full pipeline.

Features:
- Parallel processing for faster matching
- Embedding caching for improved efficiency
- Batch processing to reduce memory usage
- Thread-safe database connections for concurrent operations
- Database reset functionality to start with a clean state

Usage:
python optimized_app.py

This application only runs the full pipeline with Gemma3:4b model.
"""

import config
from agents.scheduler_agent import SchedulerAgent
from agents.cv_processor import CVProcessorAgent
from agents.jd_summarizer import JDSummarizerAgent
from efficient_matcher import EfficientMatcher
import logging
import os
import sys
import argparse
import time
from pathlib import Path
from tqdm import tqdm

# Import reset_db module for database reset functionality
try:
    import reset_db
except ImportError:
    print("Warning: Could not import reset_db module. Database reset functionality may not be available.")

# Configure logging - File Handler for all logs
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Create a custom handler for terminal output that shows all agent messages


class VerboseStreamHandler(logging.StreamHandler):
    def emit(self, record):
        # Show all log messages to terminal
        super().emit(record)


# Configure StreamHandler for terminal output
stream_handler = VerboseStreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(
    '%(name)s - %(levelname)s - %(message)s'))

# Set up root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler, file_handler]
)

logger = logging.getLogger(__name__)


class OptimizedJobScreeningApp:
    """Optimized main application class that orchestrates the job screening process."""

    def __init__(self, use_parallel=True, max_workers=4, batch_size=10, use_llm_for_top_k=5):
        """Initialize the application with performance optimizations.

        Args:
            use_parallel (bool): Whether to use parallel processing
            max_workers (int): Maximum number of parallel worker threads
            batch_size (int): Batch size for processing
            use_llm_for_top_k (int): Number of top matches to enhance with LLM
        """
        self.jd_agent = JDSummarizerAgent()
        self.cv_agent = CVProcessorAgent()
        self.scheduler_agent = SchedulerAgent()

        # Instead of using MatchingAgent, use our efficient implementation
        self.efficient_matcher = EfficientMatcher(
            use_parallel=use_parallel,
            max_workers=max_workers,
            batch_size=batch_size,
            use_llm_for_top_k=use_llm_for_top_k
        )

        logger.info(
            f"Initialized OptimizedJobScreeningApp with parallel={use_parallel}, workers={max_workers}, batch_size={batch_size}")

    def close(self):
        """Close all agents and connections."""
        self.jd_agent.close()
        self.cv_agent.close()
        self.efficient_matcher.close()
        self.scheduler_agent.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def process_job_descriptions(self, csv_path=None):
        """Process job descriptions from a CSV file.

        Args:
            csv_path (str, optional): Path to the CSV file containing job descriptions

        Returns:
            list: List of processed job IDs
        """
        logger.info("Processing job descriptions...")
        start_time = time.time()
        job_ids = self.jd_agent.process_job_descriptions_from_csv(csv_path)
        elapsed = time.time() - start_time
        logger.info(
            f"Processed {len(job_ids)} job descriptions in {elapsed:.2f} seconds")

        return job_ids

    def process_cvs(self, cv_dir=None):
        """Process CV files.

        Args:
            cv_dir (str, optional): Directory containing CV files

        Returns:
            list: List of processed candidate IDs
        """
        logger.info("Processing CVs...")
        start_time = time.time()
        candidate_ids = self.cv_agent.process_cv_directory(cv_dir)
        elapsed = time.time() - start_time
        logger.info(
            f"Processed {len(candidate_ids)} CVs in {elapsed:.2f} seconds")

        return candidate_ids

    def precompute_embeddings(self, job_ids=None, candidate_ids=None):
        """Precompute and cache embeddings for all jobs and candidates.

        Args:
            job_ids (list, optional): List of job IDs or None for all jobs
            candidate_ids (list, optional): List of candidate IDs or None for all candidates

        Returns:
            tuple: (job_embeddings_count, candidate_embeddings_count)
        """
        logger.info(
            "Precomputing embeddings (this may take a while but will speed up matching)...")
        start_time = time.time()

        # Precompute job embeddings
        job_embeddings = self.efficient_matcher.precompute_job_embeddings(
            job_ids)

        # Precompute candidate embeddings
        candidate_embeddings = self.efficient_matcher.precompute_candidate_embeddings(
            candidate_ids)

        elapsed = time.time() - start_time
        logger.info(
            f"Precomputed embeddings for {len(job_embeddings)} jobs and {len(candidate_embeddings)} candidates in {elapsed:.2f} seconds")

        return (len(job_embeddings), len(candidate_embeddings))

    def match_jobs_with_candidates(self, job_ids=None, candidate_ids=None, enhance_top_k=None):
        """Match jobs with candidates using the efficient matcher.

        Args:
            job_ids (list, optional): List of job IDs to match. If None, match all jobs.
            candidate_ids (list, optional): List of candidate IDs or None for all candidates
            enhance_top_k (int, optional): Number of top matches to enhance with LLM

        Returns:
            dict: Dictionary mapping job IDs to lists of match results
        """
        logger.info("Efficiently matching jobs with candidates...")
        start_time = time.time()

        # Use efficient matcher
        all_results = self.efficient_matcher.match_jobs_with_candidates(
            job_ids=job_ids,
            candidate_ids=candidate_ids,
            enhance_top_k=enhance_top_k,
            save_to_db=True
        )

        elapsed = time.time() - start_time
        job_count = len(all_results)
        total_matches = sum(len(matches) for matches in all_results.values())

        logger.info(
            f"Efficiently matched {job_count} jobs with candidates ({total_matches} total matches) in {elapsed:.2f} seconds")
        logger.info(
            f"Average time per job: {elapsed/max(1, job_count):.2f} seconds")

        return all_results

    def shortlist_candidates(self, job_ids=None):
        """Shortlist candidates for jobs based on match scores.

        Args:
            job_ids (list, optional): List of job IDs to shortlist for. If None, shortlist for all jobs.

        Returns:
            dict: Dictionary mapping job IDs to the number of shortlisted candidates
        """
        logger.info("Shortlisting candidates...")
        start_time = time.time()

        shortlist_counts = {}

        if job_ids:
            for job_id in tqdm(job_ids, desc="Shortlisting candidates for jobs"):
                count = self.efficient_matcher.process_shortlisting(job_id)
                shortlist_counts[job_id] = count

            logger.info(f"Shortlisted candidates for {len(job_ids)} jobs")
        else:
            # Get all jobs
            jobs = self.jd_agent.get_all_job_summaries()

            for job in tqdm(jobs, desc="Shortlisting candidates for all jobs"):
                job_id = job['id']
                count = self.efficient_matcher.process_shortlisting(job_id)
                shortlist_counts[job_id] = count

            logger.info(f"Shortlisted candidates for all jobs")

        elapsed = time.time() - start_time
        logger.info(f"Shortlisting completed in {elapsed:.2f} seconds")
        return shortlist_counts

    def schedule_interviews(self, job_ids=None):
        """Schedule interviews for shortlisted candidates.

        Args:
            job_ids (list, optional): List of job IDs to schedule interviews for. If None, schedule for all jobs.

        Returns:
            dict: Dictionary mapping job IDs to the number of scheduled interviews
        """
        logger.info("Scheduling interviews...")
        start_time = time.time()

        if job_ids:
            interview_counts = {}
            for job_id in tqdm(job_ids, desc="Scheduling interviews for jobs"):
                count = self.scheduler_agent.schedule_interviews_for_job(
                    job_id)
                interview_counts[job_id] = count

            logger.info(
                f"Scheduled interviews for shortlisted candidates of {len(job_ids)} jobs")

            elapsed = time.time() - start_time
            logger.info(
                f"Interview scheduling completed in {elapsed:.2f} seconds")
            return interview_counts
        else:
            total_interviews = self.scheduler_agent.schedule_all_pending_interviews()

            elapsed = time.time() - start_time
            logger.info(
                f"Scheduled a total of {total_interviews} interviews across all jobs in {elapsed:.2f} seconds")
            return {"total": total_interviews}

    def run_full_pipeline(self, csv_path=None, cv_dir=None, precompute=True, enhance_top_k=5):
        """Run the full job screening pipeline with optimizations.

        Args:
            csv_path (str, optional): Path to the CSV file containing job descriptions
            cv_dir (str, optional): Directory containing CV files
            precompute (bool): Whether to precompute all embeddings in advance
            enhance_top_k (int): Number of top matches per job to enhance with LLM

        Returns:
            dict: Summary of the pipeline results
        """
        total_start_time = time.time()
        logger.info("Starting optimized job screening pipeline...")

        # Process job descriptions
        job_ids = self.process_job_descriptions(csv_path)

        # Process CVs
        candidate_ids = self.process_cvs(cv_dir)

        # Precompute embeddings if requested
        if precompute:
            self.precompute_embeddings(job_ids, candidate_ids)

        # Match jobs with candidates using efficient matcher
        match_results = self.match_jobs_with_candidates(
            job_ids=job_ids,
            candidate_ids=candidate_ids,
            enhance_top_k=enhance_top_k
        )

        # Shortlist candidates
        shortlist_counts = self.shortlist_candidates(job_ids)

        # Schedule interviews
        interview_counts = self.schedule_interviews(job_ids)

        # Create a summary of the pipeline results
        total_elapsed = time.time() - total_start_time
        summary = {
            "jobs_processed": len(job_ids),
            "cvs_processed": len(candidate_ids),
            "shortlisted_candidates": shortlist_counts,
            "interviews_scheduled": interview_counts,
            "total_time_seconds": total_elapsed,
            "avg_time_per_job_seconds": total_elapsed / max(1, len(job_ids))
        }

        logger.info(
            f"Completed full optimized job screening pipeline in {total_elapsed:.2f} seconds")
        return summary


def reset_database(confirm=False):
    """Reset the database to a clean state and reimport job descriptions.

    Args:
        confirm (bool): Whether to automatically confirm the reset without prompting

    Returns:
        bool: True if reset was successful, False otherwise
    """
    try:
        logger.info("Resetting database to a clean state...")

        # If not auto-confirming, we need to mimic the reset_db.py behavior
        if not confirm:
            print("\n=== DATABASE RESET AND JOB DESCRIPTIONS REIMPORT ===")
            print("This will reset the database, deleting all data.")
            print(
                "All existing data (candidates, matches, interviews, emails) will be deleted.")

            user_response = input(
                "\nAre you sure you want to proceed? (yes/no): ").strip().lower()
            if user_response != "yes":
                logger.info("User cancelled database reset operation")
                return False

        # Reset database tables (passing True to auto-confirm the operation)
        if hasattr(reset_db, 'reset_database'):
            # We need to check if the function expects a confirm parameter
            import inspect
            sig = inspect.signature(reset_db.reset_database)
            if 'confirm' in sig.parameters:
                # The function accepts a confirm parameter
                success = reset_db.reset_database(confirm=True)
            else:
                # The function doesn't have a confirm parameter, call it without
                success = reset_db.reset_database()

            if not success:
                logger.error("Failed to reset database tables")
                return False

            logger.info("Successfully reset database tables")
        else:
            logger.error(
                "reset_database function not found in reset_db module")
            return False

        # Clean generated emails
        if hasattr(reset_db, 'clean_generated_emails'):
            reset_db.clean_generated_emails()
            logger.info("Cleaned generated emails")
        else:
            logger.warning(
                "clean_generated_emails function not found in reset_db module")

        # Re-import initial job descriptions
        if hasattr(reset_db, 'reimport_job_descriptions'):
            if reset_db.reimport_job_descriptions():
                logger.info("Successfully reimported job descriptions")
                return True
            else:
                logger.error("Failed to reimport job descriptions")
                return False
        else:
            logger.error(
                "reimport_job_descriptions function not found in reset_db module")
            return False

    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        return False


def main():
    """Main entry point for the optimized application."""
    # Set paths
    jd_path = config.JD_CSV_PATH
    cv_dir = config.CV_DIR
    use_parallel = True
    precompute = True
    top_k = 5
    workers = 8
    batch_size = 20

    # Display welcome message
    print("\n===== Optimized AI-Powered Job Application Screening System =====")
    print(f"JD Path: {jd_path}")
    print(f"CV Directory: {cv_dir}")
    print(f"Running full pipeline with Gemma3:4b model")
    print("==============================================================\n")

    # Reset database
    print("[SYSTEM] Resetting database...")
    # Auto-confirm to avoid prompting
    if not reset_database(confirm=True):
        print("[SYSTEM] Warning: Database reset failed or was incomplete.")
        print("[SYSTEM] You may want to run 'python reset_db.py' manually before proceeding.")
        user_response = input("[SYSTEM] Do you want to continue anyway? (y/n): ")
        if user_response.lower() != 'y':
            print("[SYSTEM] Exiting...")
            return
    else:
        print("[SYSTEM] Database reset completed successfully")

    with OptimizedJobScreeningApp(
        use_parallel=use_parallel,
        max_workers=workers,
        batch_size=batch_size,
        use_llm_for_top_k=top_k
    ) as app:
        print("[SYSTEM] Running full optimized pipeline...")
        summary = app.run_full_pipeline(
            csv_path=jd_path,
            cv_dir=cv_dir,
            precompute=precompute,
            enhance_top_k=top_k
        )

        # Print summary
        print("\n===== Job Screening Summary =====")
        print(f"Jobs processed: {summary['jobs_processed']}")
        print(f"CVs processed: {summary['cvs_processed']}")
        print(f"Total shortlisted candidates: {sum(summary['shortlisted_candidates'].values())}")

        if 'total' in summary['interviews_scheduled']:
            print(f"Total interviews scheduled: {summary['interviews_scheduled']['total']}")
        else:
            print(f"Total interviews scheduled: {sum(summary['interviews_scheduled'].values())}")

        print(f"Total processing time: {summary['total_time_seconds']:.2f} seconds")
        print(f"Average time per job: {summary['avg_time_per_job_seconds']:.2f} seconds")
        print("===============================\n")


if __name__ == "__main__":
    main()
