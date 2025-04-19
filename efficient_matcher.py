"""
Efficient Matcher

This module provides optimized job-candidate matching functionality with performance enhancements:
- Pre-computation and caching of embeddings
- Batch processing
- Parallelization
- Progress reporting
"""

import config
from utils.ollama_client import OllamaClient
from utils.embeddings import EmbeddingUtility
from database.db_manager import DBManager
import logging
import time
import concurrent.futures
import numpy as np
import os
import pickle
from tqdm import tqdm
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))


logger = logging.getLogger(__name__)


class EfficientMatcher:
    """Efficient matching system for job-candidate matching."""

    def __init__(self, embeddings_cache_dir=None, use_parallel=True, max_workers=4, batch_size=10,
                 use_llm_for_top_k=20, llm_quality_vs_speed=0.7):
        """Initialize the efficient matcher.

        Args:
            embeddings_cache_dir (str, optional): Directory to cache embeddings
            use_parallel (bool, optional): Whether to use parallel processing
            max_workers (int, optional): Maximum number of parallel workers
            batch_size (int, optional): Batch size for processing
            use_llm_for_top_k (int, optional): Number of top matches to generate LLM notes for
            llm_quality_vs_speed (float, optional): Tradeoff between quality and speed (0-1)
        """
        # This is the main DB connection that should be used only in the main thread
        self.db_manager = DBManager()
        self.embedding_util = EmbeddingUtility()
        self.match_threshold = config.MATCH_THRESHOLD

        # Performance settings
        self.embeddings_cache_dir = embeddings_cache_dir or os.path.join(
            config.BASE_DIR, "cache", "embeddings")
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.use_llm_for_top_k = use_llm_for_top_k
        self.llm_quality_vs_speed = llm_quality_vs_speed

        # Create cache directory if it doesn't exist
        os.makedirs(self.embeddings_cache_dir, exist_ok=True)

        # Caches
        self.job_embeddings = {}  # job_id -> embedding
        self.candidate_embeddings = {}  # candidate_id -> embedding

        # LLM client (for detailed notes)
        self.ollama_client = OllamaClient()

        # Note about thread safety
        if self.use_parallel:
            logger.info(
                "Using parallel processing - thread-local database connections will be created as needed")

        logger.info(
            "Initialized EfficientMatcher with performance optimizations")

    def close(self):
        """Close resources."""
        # Close the main database connection (should only be used in the main thread)
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.close()
            self.db_manager = None

        # Close the Ollama client
        if hasattr(self, 'ollama_client') and self.ollama_client:
            self.ollama_client.close()
            self.ollama_client = None

    def _get_embedding_cache_path(self, type_prefix, item_id):
        """Get cache path for an embedding.

        Args:
            type_prefix (str): Type prefix ('job' or 'candidate')
            item_id (int): Item ID

        Returns:
            str: Cache path
        """
        return os.path.join(self.embeddings_cache_dir, f"{type_prefix}_{item_id}.pkl")

    def _load_cached_embedding(self, type_prefix, item_id):
        """Load embedding from cache.

        Args:
            type_prefix (str): Type prefix ('job' or 'candidate')
            item_id (int): Item ID

        Returns:
            numpy.ndarray: Embedding or None if not cached
        """
        cache_path = self._get_embedding_cache_path(type_prefix, item_id)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                    logger.debug(
                        f"Loaded {type_prefix} embedding from cache: {item_id}")
                    return embedding
            except Exception as e:
                logger.warning(f"Error loading cached embedding: {str(e)}")

        return None

    def _save_embedding_to_cache(self, type_prefix, item_id, embedding):
        """Save embedding to cache.

        Args:
            type_prefix (str): Type prefix ('job' or 'candidate')
            item_id (int): Item ID
            embedding (numpy.ndarray): Embedding to cache
        """
        if embedding is None:
            return

        cache_path = self._get_embedding_cache_path(type_prefix, item_id)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
                logger.debug(f"Cached {type_prefix} embedding: {item_id}")
        except Exception as e:
            logger.warning(f"Error caching embedding: {str(e)}")

    def precompute_job_embeddings(self, job_ids=None):
        """Precompute embeddings for jobs.

        Args:
            job_ids (list, optional): List of job IDs or None for all jobs

        Returns:
            dict: Dictionary of job_id -> embedding
        """
        # Get job IDs if not provided
        if job_ids is None:
            jobs = self.db_manager.db.cursor.execute(
                'SELECT id FROM job_descriptions').fetchall()
            job_ids = [j['id'] for j in jobs]

        # Check which embeddings are already cached
        for job_id in job_ids:
            cached_embedding = self._load_cached_embedding('job', job_id)
            if cached_embedding is not None:
                self.job_embeddings[job_id] = cached_embedding

        # Compute missing embeddings
        missing_job_ids = [
            jid for jid in job_ids if jid not in self.job_embeddings]

        if missing_job_ids:
            logger.info(
                f"Computing embeddings for {len(missing_job_ids)} jobs...")

            # Process in parallel if enabled
            if self.use_parallel and len(missing_job_ids) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_job = {
                        executor.submit(self._compute_job_embedding, job_id): job_id
                        for job_id in missing_job_ids
                    }

                    # Show progress
                    for _ in tqdm(concurrent.futures.as_completed(future_to_job),
                                  total=len(missing_job_ids),
                                  desc="Computing job embeddings"):
                        pass  # Progress is tracked by tqdm
            else:
                # Sequential processing
                for job_id in tqdm(missing_job_ids, desc="Computing job embeddings"):
                    self._compute_job_embedding(job_id)

        logger.info(
            f"Job embeddings ready for {len(self.job_embeddings)} jobs")
        return self.job_embeddings

    def _compute_job_embedding(self, job_id):
        """Compute embedding for a job.

        Args:
            job_id (int): Job ID

        Returns:
            numpy.ndarray: Job embedding
        """
        try:
            # Create a thread-local DB connection to avoid SQLite thread errors
            thread_local_db = DBManager()

            # Get job data using the thread-local connection
            job_data = thread_local_db.get_job(job_id)
            if not job_data:
                logger.warning(f"Job ID {job_id} not found")
                return None

            # Prepare text for embedding
            job_text = f"Job Title: {job_data['title']}\nSkills: {', '.join(job_data['skills'])}\nDescription: {job_data['description']}"

            # Generate embedding
            embedding = self.embedding_util.get_embedding(
                job_text, use_cache=False)

            # Cache the embedding
            self.job_embeddings[job_id] = embedding
            self._save_embedding_to_cache('job', job_id, embedding)

            # Close the thread-local connection
            thread_local_db.close()

            return embedding
        except Exception as e:
            logger.error(
                f"Error computing job embedding for job {job_id}: {str(e)}")
            return None

    def precompute_candidate_embeddings(self, candidate_ids=None):
        """Precompute embeddings for candidates.

        Args:
            candidate_ids (list, optional): List of candidate IDs or None for all candidates

        Returns:
            dict: Dictionary of candidate_id -> embedding
        """
        # Get candidate IDs if not provided
        if candidate_ids is None:
            candidates = self.db_manager.db.cursor.execute(
                'SELECT id FROM candidates').fetchall()
            candidate_ids = [c['id'] for c in candidates]

        # Check which embeddings are already cached
        for candidate_id in candidate_ids:
            cached_embedding = self._load_cached_embedding(
                'candidate', candidate_id)
            if cached_embedding is not None:
                self.candidate_embeddings[candidate_id] = cached_embedding

        # Compute missing embeddings
        missing_candidate_ids = [
            cid for cid in candidate_ids if cid not in self.candidate_embeddings]

        if missing_candidate_ids:
            logger.info(
                f"Computing embeddings for {len(missing_candidate_ids)} candidates...")

            # Process in parallel if enabled
            if self.use_parallel and len(missing_candidate_ids) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_candidate = {
                        executor.submit(self._compute_candidate_embedding, candidate_id): candidate_id
                        for candidate_id in missing_candidate_ids
                    }

                    # Show progress
                    for _ in tqdm(concurrent.futures.as_completed(future_to_candidate),
                                  total=len(missing_candidate_ids),
                                  desc="Computing candidate embeddings"):
                        pass  # Progress is tracked by tqdm
            else:
                # Sequential processing
                for candidate_id in tqdm(missing_candidate_ids, desc="Computing candidate embeddings"):
                    self._compute_candidate_embedding(candidate_id)

        logger.info(
            f"Candidate embeddings ready for {len(self.candidate_embeddings)} candidates")
        return self.candidate_embeddings

    def _compute_candidate_embedding(self, candidate_id):
        """Compute embedding for a candidate.

        Args:
            candidate_id (int): Candidate ID

        Returns:
            numpy.ndarray: Candidate embedding
        """
        try:
            # Create a thread-local DB connection to avoid SQLite thread errors
            thread_local_db = DBManager()

            # Get candidate data using the thread-local connection
            candidate_data = thread_local_db.get_candidate(candidate_id)
            if not candidate_data:
                logger.warning(f"Candidate ID {candidate_id} not found")
                return None

            # Prepare text for embedding
            candidate_text = f"Skills: {', '.join(candidate_data['skills'])}\nQualifications: {', '.join(candidate_data['qualifications'])}\nExperience: {'; '.join(candidate_data['experience'])}"

            # Generate embedding
            embedding = self.embedding_util.get_embedding(
                candidate_text, use_cache=False)

            # Cache the embedding
            self.candidate_embeddings[candidate_id] = embedding
            self._save_embedding_to_cache('candidate', candidate_id, embedding)

            # Close the thread-local connection
            thread_local_db.close()

            return embedding
        except Exception as e:
            logger.error(
                f"Error computing candidate embedding for candidate {candidate_id}: {str(e)}")
            return None

    def compute_initial_matches(self, job_id, candidate_ids=None, top_k=None):
        """Compute initial matches for a job using fast embedding similarity.

        Args:
            job_id (int): Job ID
            candidate_ids (list, optional): List of candidate IDs or None for all candidates
            top_k (int, optional): Number of top matches to return

        Returns:
            list: List of match results sorted by score (descending)
        """
        # Get all candidates if not provided
        if candidate_ids is None:
            # Use a copy of the cursor to avoid thread safety issues
            thread_local_db = DBManager()
            candidates = thread_local_db.db.cursor.execute(
                'SELECT id FROM candidates').fetchall()
            candidate_ids = [c['id'] for c in candidates]
            thread_local_db.close()

        # Ensure we have job embedding
        if job_id not in self.job_embeddings:
            self.precompute_job_embeddings([job_id])

        # Ensure we have candidate embeddings
        missing_candidates = [
            cid for cid in candidate_ids if cid not in self.candidate_embeddings]
        if missing_candidates:
            self.precompute_candidate_embeddings(missing_candidates)

        # Get job embedding
        job_embedding = self.job_embeddings.get(job_id)
        if job_embedding is None:
            logger.error(f"No embedding available for job {job_id}")
            return []

        # Compute similarities and create match results
        match_results = []

        # Create a thread-local database connection for getting candidate names
        thread_local_db = DBManager()

        for candidate_id in candidate_ids:
            candidate_embedding = self.candidate_embeddings.get(candidate_id)
            if candidate_embedding is None:
                continue

            # Compute cosine similarity
            similarity = float(np.dot(job_embedding, candidate_embedding) /
                               (np.linalg.norm(job_embedding) * np.linalg.norm(candidate_embedding)))

            # Get candidate data for basic info
            candidate_data = thread_local_db.get_candidate(candidate_id)

            # Add to results
            match_results.append({
                'job_id': job_id,
                'candidate_id': candidate_id,
                'candidate_name': candidate_data.get('name', '') if candidate_data else '',
                'similarity_score': similarity,
                'overall_match': similarity,  # Use similarity as initial overall match
                'has_llm_notes': False
            })

        # Close the thread-local database connection
        thread_local_db.close()

        # Sort by score (descending)
        match_results.sort(key=lambda x: x['similarity_score'], reverse=True)

        # Limit to top_k if specified
        if top_k is not None and top_k > 0:
            match_results = match_results[:top_k]

        return match_results

    def enhance_matches_with_llm(self, job_id, initial_matches, max_llm_matches=None):
        """Enhance match results with LLM-generated notes and analysis.

        Args:
            job_id (int): Job ID
            initial_matches (list): List of initial match results
            max_llm_matches (int, optional): Maximum number of matches to enhance with LLM

        Returns:
            list: Enhanced match results
        """
        # Limit number of LLM calls if specified
        if max_llm_matches is None:
            max_llm_matches = self.use_llm_for_top_k

        matches_to_enhance = initial_matches[:max_llm_matches]

        # Get job data
        job_data = self.db_manager.get_job(job_id)
        if not job_data:
            logger.error(f"Job ID {job_id} not found")
            return initial_matches

        # Process in parallel if enabled and we have multiple matches
        if self.use_parallel and len(matches_to_enhance) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_match = {
                    executor.submit(self._enhance_match_with_llm, job_data, match): i
                    for i, match in enumerate(matches_to_enhance)
                }

                # Show progress
                for future in tqdm(concurrent.futures.as_completed(future_to_match),
                                   total=len(matches_to_enhance),
                                   desc="Enhancing matches with LLM"):
                    idx = future_to_match[future]
                    try:
                        enhanced_match = future.result()
                        if enhanced_match:
                            matches_to_enhance[idx] = enhanced_match
                    except Exception as e:
                        logger.error(f"Error enhancing match: {str(e)}")
        else:
            # Sequential processing
            for i, match in enumerate(tqdm(matches_to_enhance, desc="Enhancing matches with LLM")):
                enhanced_match = self._enhance_match_with_llm(job_data, match)
                if enhanced_match:
                    matches_to_enhance[i] = enhanced_match

        # Combine enhanced matches with remaining matches
        if max_llm_matches < len(initial_matches):
            remaining_matches = initial_matches[max_llm_matches:]
            return matches_to_enhance + remaining_matches
        else:
            return matches_to_enhance

    def _enhance_match_with_llm(self, job_data, match):
        """Enhance a single match with LLM analysis.

        Args:
            job_data (dict): Job data
            match (dict): Match result

        Returns:
            dict: Enhanced match result
        """
        try:
            candidate_id = match['candidate_id']

            # Create a thread-local DB connection to avoid SQLite thread errors
            thread_local_db = DBManager()

            # Get candidate data using the thread-local connection
            candidate_data = thread_local_db.get_candidate(candidate_id)
            if not candidate_data:
                logger.warning(f"Candidate ID {candidate_id} not found")
                thread_local_db.close()
                return match

            # Prepare job data
            job_summary = {
                'title': job_data['title'],
                'skills': job_data['skills'],
                'description': job_data['description'][:300] + '...' if len(job_data['description']) > 300 else job_data['description']
            }

            # Prepare candidate data
            candidate_summary = {
                'name': candidate_data['name'],
                'skills': candidate_data['skills'],
                'qualifications': candidate_data['qualifications'],
                'experience': candidate_data['experience'][:3] if len(candidate_data['experience']) > 3 else candidate_data['experience']
            }

            # Close the thread-local connection
            thread_local_db.close()

            # Prepare match scores
            scores = {
                'similarity_score': f"{match['similarity_score']:.2f}"
            }

            # Build prompt
            prompt = f"""Analyze the match between the following job and candidate:
            
            JOB:
            {json.dumps(job_summary, indent=2)}
            
            CANDIDATE:
            {json.dumps(candidate_summary, indent=2)}
            
            MATCH SCORES:
            {json.dumps(scores, indent=2)}
            
            Provide a concise analysis of why this candidate is a good match or not for this job.
            Focus on specific skills, qualifications, and experience that match or don't match.
            Limit your response to 5-6 sentences.
            """

            # Generate LLM notes
            llm_notes = self.ollama_client.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            )

            # Update match with LLM notes
            enhanced_match = match.copy()
            enhanced_match['notes'] = llm_notes
            enhanced_match['has_llm_notes'] = True

            # Add detailed scores
            # Calculate skills match
            job_skills = set([s.lower() for s in job_data['skills']])
            candidate_skills = set([s.lower()
                                   for s in candidate_data['skills']])
            skills_intersection = len(
                job_skills.intersection(candidate_skills))
            skills_union = len(job_skills.union(candidate_skills))
            skills_match = skills_intersection / skills_union if skills_union > 0 else 0

            # Simple scores for other dimensions
            quals_match = 0.7  # Placeholder
            exp_match = 0.6  # Placeholder

            # Calculate weighted overall match
            overall_match = (skills_match * 0.5) + (quals_match *
                                                    0.3) + (match['similarity_score'] * 0.2)

            # Update scores
            enhanced_match['skills_match'] = skills_match
            enhanced_match['qualifications_match'] = quals_match
            # Use similarity as experience match
            enhanced_match['experience_match'] = match['similarity_score']
            enhanced_match['overall_match'] = overall_match

            return enhanced_match

        except Exception as e:
            logger.error(f"Error enhancing match with LLM: {str(e)}")
            return match

    def _traditional_matching(self, job_id, candidate_ids, local_db=None, save_to_db=True):
        """Perform traditional matching without using LLM or embeddings.

        Args:
            job_id (int): Job ID
            candidate_ids (list): List of candidate IDs
            local_db (DBManager, optional): Thread-local database connection
            save_to_db (bool): Whether to save results to database

        Returns:
            list: List of match results
        """
        logger.info(
            f"Using traditional matching for job {job_id} (LLM and embeddings disabled)")

        # Get job data for traditional matching
        job_data = (local_db or self.db_manager).get_job_by_id(job_id)
        job_text = job_data.get('jd_text', '') + ' ' + \
            job_data.get('summary', '')

        # Get all candidates
        cursor = (local_db.db.cursor if local_db else self.db_manager.db.cursor)
        all_candidates = []

        # Process in batches
        for i in range(0, len(candidate_ids), self.batch_size):
            batch_ids = candidate_ids[i:i+self.batch_size]
            placeholders = ','.join(['?'] * len(batch_ids))
            query = f'SELECT * FROM candidates WHERE id IN ({placeholders})'

            candidates_batch = cursor.execute(query, batch_ids).fetchall()
            all_candidates.extend(candidates_batch)

        # Calculate traditional TF-IDF matches
        matches = []
        for candidate in tqdm(all_candidates, desc=f"Matching job {job_id} with candidates (traditional)"):
            candidate_text = candidate.get('text', '')

            # Calculate match score using text similarity
            match_score = self.embedding_util.calculate_text_similarity(
                job_text, candidate_text)

            # Create match result
            match_result = {
                'job_id': job_id,
                'candidate_id': candidate['id'],
                'match_score': match_score,
                'match_notes': "Match calculated using traditional text similarity only.",
                'is_shortlisted': match_score >= self.match_threshold
            }
            matches.append(match_result)

        # Sort by match score in descending order
        matches.sort(key=lambda x: x['match_score'], reverse=True)

        # Save to database if requested
        if save_to_db:
            self._save_match_results_to_db(
                matches, local_db or self.db_manager)

        return matches

    def match_job_with_all_candidates(self, job_id, candidate_ids=None, enhance_top_k=None, save_to_db=True):
        """Match a job with all candidates.

        Args:
            job_id (int): Job ID
            candidate_ids (list, optional): List of candidate IDs or None for all candidates
            enhance_top_k (int, optional): Number of top matches to enhance with LLM
            save_to_db (bool, optional): Whether to save results to database

        Returns:
            list: List of match results
        """
        start_time = time.time()
        logger.info(f"Matching job {job_id} with candidates...")

        # Get job data (for LLM enhancement)
        job_data = self.db_manager.get_job_by_id(job_id)
        if not job_data:
            logger.error(f"Job {job_id} not found in database")
            return []

        # Create a new thread-local DBManager if in parallel mode
        local_db = None
        if self.use_parallel:
            local_db = DBManager()

        try:
            # Get candidate IDs if not provided
            if candidate_ids is None:
                cursor = (
                    local_db.db.cursor if local_db else self.db_manager.db.cursor)
                candidates = cursor.execute(
                    'SELECT id FROM candidates').fetchall()
                candidate_ids = [c['id'] for c in candidates]
                logger.info(
                    f"Found {len(candidate_ids)} candidates in database")

            # Skip if no candidates
            if not candidate_ids:
                logger.warning(f"No candidates found for job {job_id}")
                return []

            # Use only traditional/faster matching if LLM and embeddings are disabled
            if not getattr(config, 'USE_LLM', True) and not getattr(config, 'USE_EMBEDDINGS', True):
                matches = self._traditional_matching(
                    job_id, candidate_ids, local_db, save_to_db)
                elapsed = time.time() - start_time
                logger.info(
                    f"Matched job {job_id} with {len(candidate_ids)} candidates in {elapsed:.2f} seconds (traditional matching only)")
                return matches

            # If embeddings are enabled but LLM is disabled
            if getattr(config, 'USE_EMBEDDINGS', True) and not getattr(config, 'USE_LLM', True):
                enhance_top_k = 0  # Don't enhance any matches with LLM

            # Default enhance top k value
            if enhance_top_k is None:
                enhance_top_k = self.use_llm_for_top_k

            # Compute initial matches using embeddings
            try:
                logger.info(f"Computing initial matches for job {job_id}")
                initial_matches = self.compute_initial_matches(
                    job_id, candidate_ids)
                logger.info(
                    f"Found {len(initial_matches)} initial matches for job {job_id}")
            except Exception as e:
                logger.error(
                    f"Error computing initial matches for job {job_id}: {str(e)}", exc_info=True)
                initial_matches = []

            # Early return if no initial matches
            if not initial_matches:
                logger.warning(f"No initial matches found for job {job_id}")
                return []

            # Enhance top matches with LLM if requested
            if getattr(config, 'USE_LLM', True) and enhance_top_k > 0 and initial_matches:
                try:
                    logger.info(
                        f"Enhancing top {enhance_top_k} matches with LLM for job {job_id}")
                    enhanced_matches = self.enhance_matches_with_llm(
                        job_id, initial_matches, enhance_top_k)
                    logger.info(
                        f"Enhanced {len(enhanced_matches[:enhance_top_k])} matches with LLM")
                except Exception as e:
                    logger.error(
                        f"Error enhancing matches with LLM: {str(e)}", exc_info=True)
                    enhanced_matches = initial_matches
            else:
                enhanced_matches = initial_matches
                if enhance_top_k > 0 and getattr(config, 'USE_LLM', True):
                    logger.warning(f"No matches to enhance for job {job_id}")
                elif not getattr(config, 'USE_LLM', True) and enhance_top_k > 0:
                    logger.info(
                        f"LLM enhancement disabled, skipping for job {job_id}")

            # Save to database if requested
            if save_to_db and enhanced_matches:
                try:
                    logger.info(
                        f"Saving {len(enhanced_matches)} match results to database")
                    self._save_match_results_to_db(
                        enhanced_matches, local_db or self.db_manager)
                except Exception as e:
                    logger.error(
                        f"Error saving match results to database: {str(e)}", exc_info=True)

            elapsed = time.time() - start_time
            logger.info(
                f"Matched job {job_id} with {len(candidate_ids)} candidates in {elapsed:.2f} seconds")
            return enhanced_matches

        except Exception as e:
            logger.error(
                f"Unexpected error in match_job_with_all_candidates for job {job_id}: {str(e)}", exc_info=True)
            return []
        finally:
            # Close the local DB connection if created
            if local_db:
                local_db.close()

    def match_jobs_with_candidates(self, job_ids=None, candidate_ids=None, enhance_top_k=None, save_to_db=True):
        """Match multiple jobs with candidates.

        Args:
            job_ids (list, optional): List of job IDs or None for all jobs
            candidate_ids (list, optional): List of candidate IDs or None for all candidates
            enhance_top_k (int, optional): Number of top matches to enhance with LLM
            save_to_db (bool, optional): Whether to save matches to the database

        Returns:
            dict: Dictionary mapping job IDs to lists of match results
        """
        # Get all job IDs if not provided
        if job_ids is None:
            # Use a thread-local database connection
            thread_local_db = DBManager()
            jobs = thread_local_db.db.cursor.execute(
                'SELECT id FROM job_descriptions').fetchall()
            job_ids = [j['id'] for j in jobs]
            thread_local_db.close()

        # Precompute all embeddings in advance
        logger.info("Precomputing embeddings for efficient matching...")
        self.precompute_job_embeddings(job_ids)
        self.precompute_candidate_embeddings(candidate_ids)

        # Match each job with candidates
        all_results = {}

        for job_id in tqdm(job_ids, desc="Matching jobs with candidates"):
            match_results = self.match_job_with_all_candidates(
                job_id=job_id,
                candidate_ids=candidate_ids,
                enhance_top_k=enhance_top_k,
                save_to_db=save_to_db
            )
            all_results[job_id] = match_results

        logger.info(f"Completed matching {len(job_ids)} jobs with candidates")
        return all_results

    def process_shortlisting(self, job_id):
        """Process shortlisting for a job based on match scores.

        Args:
            job_id (int): Job ID

        Returns:
            int: Number of shortlisted candidates
        """
        # Create a thread-local database connection
        thread_local_db = DBManager()
        count = thread_local_db.process_shortlisting(job_id)
        thread_local_db.close()
        return count

    def _save_match_results_to_db(self, matches, db_manager):
        """Save match results to database.

        Args:
            matches (list): List of match results
            db_manager (DBManager): Database manager to use

        Returns:
            int: Number of matches saved
        """
        saved_count = 0

        # Process in batches for better performance
        for i in range(0, len(matches), self.batch_size):
            batch = matches[i:i+self.batch_size]

            for match in batch:
                # Get match fields with defaults for backward compatibility
                job_id = match['job_id']
                candidate_id = match['candidate_id']

                # Different methods might provide different score fields
                match_score = match.get(
                    'overall_match', match.get('match_score', 0))
                skills_match = match.get(
                    'skills_match', match.get('similarity_score', 0.5))
                quals_match = match.get('qualifications_match', 0.5)
                exp_match = match.get('experience_match', 0.5)
                notes = match.get('notes', match.get(
                    'match_notes', f"Match score: {match_score:.2f}"))

                # Save match result to database
                match_id = db_manager.save_match_result(
                    job_id=job_id,
                    candidate_id=candidate_id,
                    match_score=match_score,
                    skills_match=skills_match,
                    quals_match=quals_match,
                    exp_match=exp_match,
                    notes=notes
                )

                # Add match ID to result for reference
                match['id'] = match_id
                saved_count += 1

        logger.info(f"Saved {saved_count} matches to database")
        return saved_count


# Example usage
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
    print("EFFICIENT MATCHER - DEMONSTRATION")
    print("=" * 50)

    matcher = EfficientMatcher(
        use_parallel=True,
        max_workers=4,
        batch_size=10,
        use_llm_for_top_k=5  # Only use LLM for top 5 matches per job
    )

    try:
        # Get a job
        job = matcher.db_manager.db.cursor.execute(
            'SELECT id FROM job_descriptions LIMIT 1').fetchone()

        if job:
            job_id = job['id']

            print(f"\nMatching job {job_id} with all candidates...")
            matches = matcher.match_job_with_all_candidates(job_id)

            print(f"\nTop matches for job {job_id}:")
            for i, match in enumerate(matches[:5]):
                print(f"\n{i+1}. Candidate: {match.get('candidate_name', 'ID ' + str(match.get('candidate_id')))} - Score: {match.get('overall_match', 0):.2f}")
                print(f"   Has LLM notes: {match.get('has_llm_notes', False)}")
                print(
                    f"   Notes: {match.get('notes', 'No notes available')[:100]}...")
        else:
            print("No jobs found in the database. Please populate the database first.")
    finally:
        matcher.close()
