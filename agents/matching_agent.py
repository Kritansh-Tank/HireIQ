"""
Matching Agent

This agent compares job descriptions with candidate CVs and calculates match scores.
It identifies the most suitable candidates for each job based on skills, qualifications,
and experience matches.
"""

import logging
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from database.db_manager import DBManager
from agents.base_agent import BaseAgent
from tools.ml_model import MLModelTool
from utils.embeddings import EmbeddingUtility
import config

logger = logging.getLogger(__name__)

class MatchingAgent(BaseAgent):
    """Agent for matching job descriptions with candidate CVs."""
    
    def __init__(self):
        """Initialize the Matching agent."""
        super().__init__(
            name="MatchingAgent",
            system_prompt="""You are an AI assistant specialized in matching job candidates with job descriptions.
            Focus on identifying the best matches based on skills, qualifications, and experience.
            Provide clear explanations for why candidates are good matches for specific roles."""
        )
        
        self.db_manager = DBManager()
        self.match_threshold = config.MATCH_THRESHOLD
        
        # Initialize ML model tool
        self.ml_tool = MLModelTool()
        
        # Register additional tools
        self._register_matching_tools()
        
        # Start message processing
        self.start_message_processing()
        
        logger.info("Initialized Matching Agent with Ollama LLM integration")
    
    def _register_matching_tools(self):
        """Register tools specific to the matching agent."""
        # ML model tools
        self.register_tool("match_job_candidates", self.ml_tool.match_job_candidates, 
                         "Match job with candidates using ML models")
        self.register_tool("extract_skills", self.ml_tool.extract_skills,
                         "Extract skills from text using ML models")
        self.register_tool("analyze_job_requirements", self.ml_tool.analyze_job_requirements,
                         "Analyze job requirements from job description")
    
    def close(self):
        """Close database connections and free resources."""
        self.db_manager.close()
        self.ml_tool.close()
        super().close()
    
    def match_job_with_candidate(self, job_id, candidate_id):
        """Match a job with a candidate and calculate match scores.
        
        Args:
            job_id (int): Job ID in the database
            candidate_id (int): Candidate ID in the database
            
        Returns:
            dict: Match results including scores
        """
        # Get job and candidate data
        job_data = self.db_manager.get_job(job_id)
        candidate_data = self.db_manager.get_candidate(candidate_id)
        
        if not job_data or not candidate_data:
            logger.warning(f"Job ID {job_id} or Candidate ID {candidate_id} not found")
            return None
        
        # Combine approaches: traditional text matching + ML + LLM
        match_results = self._calculate_match_with_combined_approach(job_data, candidate_data)
        
        # Determine if candidate should be shortlisted
        shortlisted = match_results['overall_match'] >= self.match_threshold
        
        # Add notes using LLM to explain the match
        notes = self._generate_llm_match_notes(job_data, candidate_data, match_results)
        
        # Save match results to database
        match_id = self.db_manager.save_match_result(
            job_id=job_id,
            candidate_id=candidate_id,
            match_score=match_results['overall_match'],
            skills_match=match_results['skills_match'],
            quals_match=match_results['qualifications_match'],
            exp_match=match_results['experience_match'],
            notes=notes
        )
        
        # Add match ID to results
        match_results['id'] = match_id
        match_results['shortlisted'] = shortlisted
        match_results['notes'] = notes
        
        logger.info(f"Match score for Job {job_id} and Candidate {candidate_id}: "
                   f"{match_results['overall_match']:.2f} (Shortlisted: {shortlisted})")
        
        return match_results
    
    def _calculate_match_with_combined_approach(self, job_data, candidate_data):
        """Calculate match scores using a combined approach.
        
        Args:
            job_data (dict): Job data
            candidate_data (dict): Candidate data
            
        Returns:
            dict: Match scores
        """
        # Approach 1: Traditional text matching
        # This is a simplified approach - in a real system, you would use more sophisticated matching
        
        # Skills match
        job_skills = set([s.lower() for s in job_data['skills']])
        candidate_skills = set([s.lower() for s in candidate_data['skills']])
        
        # Calculate Jaccard similarity for skills
        skills_intersection = len(job_skills.intersection(candidate_skills))
        skills_union = len(job_skills.union(candidate_skills))
        skills_match = skills_intersection / skills_union if skills_union > 0 else 0
        
        # Qualifications match (simplified)
        quals_match = 0.7  # Default value
        
        # Experience match (simplified)
        exp_match = 0.6  # Default value
        
        # Approach 2: ML model matching
        try:
            # Prepare text for ML matching
            job_text = f"Job Title: {job_data['title']}\nSkills: {', '.join(job_data['skills'])}\nDescription: {job_data['description']}"
            candidate_text = f"Skills: {', '.join(candidate_data['skills'])}\nQualifications: {', '.join(candidate_data['qualifications'])}\nExperience: {'; '.join(candidate_data['experience'])}"
            
            # Use ML tool for matching
            ml_matches = self.ml_tool.match_job_candidates(
                job_description=job_text,
                candidate_descriptions=[candidate_text],
                threshold=0.0,  # Set to 0 to get a score regardless of threshold
                top_k=1
            )
            
            # Extract ML match score if available
            ml_score = ml_matches[0][1] if ml_matches else 0.5
            
            # Blend scores (giving more weight to ML score)
            blended_skills_match = (skills_match * 0.4) + (ml_score * 0.6)
            
            # Update skills match
            skills_match = blended_skills_match
            
        except Exception as e:
            logger.error(f"Error in ML matching: {str(e)}")
            # Keep original skills match if ML fails
        
        # Approach 3: Semantic similarity using embeddings
        try:
            # Calculate semantic similarity between job and candidate
            job_emb = self.embedding_util.get_embedding(job_data['description'])
            candidate_text = f"{', '.join(candidate_data['skills'])}\n{', '.join(candidate_data['qualifications'])}\n{'; '.join(candidate_data['experience'])}"
            candidate_emb = self.embedding_util.get_embedding(candidate_text)
            
            if job_emb is not None and candidate_emb is not None:
                # Import numpy for dot product
                import numpy as np
                
                # Calculate cosine similarity
                semantic_similarity = float(np.dot(job_emb, candidate_emb) / 
                                          (np.linalg.norm(job_emb) * np.linalg.norm(candidate_emb)))
                
                # Blend with experience match
                exp_match = (exp_match * 0.4) + (semantic_similarity * 0.6)
                
        except Exception as e:
            logger.error(f"Error in semantic matching: {str(e)}")
            # Keep original experience match if semantic matching fails
        
        # Calculate overall match
        overall_match = (skills_match * 0.5) + (quals_match * 0.3) + (exp_match * 0.2)
        
        # Prepare match results
        match_results = {
            'skills_match': skills_match,
            'qualifications_match': quals_match,
            'experience_match': exp_match,
            'overall_match': overall_match,
            'semantic_similarity': semantic_similarity if 'semantic_similarity' in locals() else None,
            'ml_score': ml_score if 'ml_score' in locals() else None
        }
        
        return match_results
    
    def _generate_llm_match_notes(self, job_data, candidate_data, match_results):
        """Generate notes explaining the match results using the LLM.
        
        Args:
            job_data (dict): Job data
            candidate_data (dict): Candidate data
            match_results (dict): Match scores
            
        Returns:
            str: Notes explaining the match
        """
        try:
            # Prepare job data
            job_summary = {
                'title': job_data['title'],
                'skills': job_data['skills'],
                'description': job_data['description'][:300] + '...'  # Truncate for prompt size
            }
            
            # Prepare candidate data
            candidate_summary = {
                'name': candidate_data['name'],
                'skills': candidate_data['skills'],
                'qualifications': candidate_data['qualifications'],
                'experience': candidate_data['experience'][:3] if len(candidate_data['experience']) > 3 else candidate_data['experience']
            }
            
            # Prepare match scores
            scores = {
                'overall_match': f"{match_results['overall_match']:.2f}",
                'skills_match': f"{match_results['skills_match']:.2f}",
                'qualifications_match': f"{match_results['qualifications_match']:.2f}",
                'experience_match': f"{match_results['experience_match']:.2f}"
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
            
            # Get LLM response
            notes = self.process_with_llm(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            )
            
            # Fall back to basic notes if LLM fails
            if not notes or "Error:" in notes:
                logger.warning("Failed to generate LLM match notes, falling back to basic notes")
                return self._generate_basic_match_notes(job_data, candidate_data, match_results)
            
            return notes
            
        except Exception as e:
            logger.error(f"Error generating LLM match notes: {str(e)}")
            return self._generate_basic_match_notes(job_data, candidate_data, match_results)
    
    def _generate_basic_match_notes(self, job_data, candidate_data, match_results):
        """Generate basic notes explaining the match results (fallback method).
        
        Args:
            job_data (dict): Job data
            candidate_data (dict): Candidate data
            match_results (dict): Match scores
            
        Returns:
            str: Notes explaining the match
        """
        notes = []
        
        # Add overall score
        notes.append(f"Overall match score: {match_results['overall_match']:.2f}")
        
        # Check skills match
        skills_match = match_results['skills_match']
        notes.append(f"Skills match: {skills_match:.2f}")
        
        if skills_match > 0.8:
            notes.append("Strong skills match!")
        elif skills_match > 0.5:
            notes.append("Good skills match")
        else:
            notes.append("Limited skills match")
        
        # List matching skills
        matching_skills = set([s.lower() for s in job_data['skills']]).intersection(
            set([s.lower() for s in candidate_data['skills']]))
        if matching_skills:
            notes.append(f"Matching skills: {', '.join(list(matching_skills)[:5])}")
        
        # Add qualifications match
        quals_match = match_results['qualifications_match']
        notes.append(f"Qualifications match: {quals_match:.2f}")
        
        # Add experience match
        exp_match = match_results['experience_match']
        notes.append(f"Experience match: {exp_match:.2f}")
        
        return "\n".join(notes)
    
    def match_job_with_all_candidates(self, job_id):
        """Match a job with all candidates in the database.
        
        Args:
            job_id (int): Job ID in the database
            
        Returns:
            list: List of match results
        """
        # Get all candidates
        candidates = self.db_manager.db.cursor.execute('SELECT id FROM candidates').fetchall()
        candidate_ids = [c['id'] for c in candidates]
        
        match_results = []
        for candidate_id in candidate_ids:
            result = self.match_job_with_candidate(job_id, candidate_id)
            if result:
                match_results.append(result)
        
        # Sort by match score (descending)
        match_results.sort(key=lambda x: x['overall_match'], reverse=True)
        
        logger.info(f"Matched Job {job_id} with {len(match_results)} candidates")
        return match_results
    
    def match_all_jobs_with_all_candidates(self):
        """Match all jobs with all candidates in the database.
        
        Returns:
            dict: Dictionary mapping job IDs to lists of match results
        """
        # Get all jobs
        jobs = self.db_manager.db.cursor.execute('SELECT id FROM job_descriptions').fetchall()
        job_ids = [j['id'] for j in jobs]
        
        all_results = {}
        for job_id in job_ids:
            match_results = self.match_job_with_all_candidates(job_id)
            all_results[job_id] = match_results
        
        logger.info(f"Completed matching {len(job_ids)} jobs with all candidates")
        return all_results
    
    def get_match_results(self, job_id, shortlisted_only=False):
        """Get match results for a job.
        
        Args:
            job_id (int): Job ID in the database
            shortlisted_only (bool): Whether to return only shortlisted candidates
            
        Returns:
            list: List of match results
        """
        return self.db_manager.get_match_results(job_id, shortlisted_only)
    
    def update_shortlist_status(self, match_id, shortlisted):
        """Update the shortlisting status of a match.
        
        Args:
            match_id (int): Match ID in the database
            shortlisted (bool): Whether the candidate should be shortlisted
            
        Returns:
            bool: Success status
        """
        return self.db_manager.update_shortlist(match_id, shortlisted)
    
    def process_shortlisting(self, job_id):
        """Process shortlisting for a job based on match scores.
        
        Args:
            job_id (int): Job ID in the database
            
        Returns:
            int: Number of shortlisted candidates
        """
        return self.db_manager.process_shortlisting(job_id)
    
    def match_candidate_with_all_jobs(self, candidate_id):
        """Match a candidate with all jobs in the database.
        
        Args:
            candidate_id (int): Candidate ID in the database
            
        Returns:
            list: List of match results
        """
        # Get all jobs
        jobs = self.db_manager.db.cursor.execute('SELECT id FROM job_descriptions').fetchall()
        job_ids = [j['id'] for j in jobs]
        
        match_results = []
        for job_id in job_ids:
            result = self.match_job_with_candidate(job_id, candidate_id)
            if result:
                # Add job title to result for convenience
                job_data = self.db_manager.get_job(job_id)
                if job_data:
                    result['job_title'] = job_data['title']
                
                match_results.append(result)
        
        # Sort by match score (descending)
        match_results.sort(key=lambda x: x['overall_match'], reverse=True)
        
        logger.info(f"Matched Candidate {candidate_id} with {len(match_results)} jobs")
        return match_results
    
    def match_all_candidates_with_all_jobs(self):
        """Match all candidates with all jobs in the database.
        
        Returns:
            dict: Dictionary mapping candidate IDs to lists of match results
        """
        # Get all candidates
        candidates = self.db_manager.db.cursor.execute('SELECT id FROM candidates').fetchall()
        candidate_ids = [c['id'] for c in candidates]
        
        all_results = {}
        for candidate_id in candidate_ids:
            match_results = self.match_candidate_with_all_jobs(candidate_id)
            all_results[candidate_id] = match_results
        
        logger.info(f"Completed matching {len(candidate_ids)} candidates with all jobs")
        return all_results
    
    def handle_message(self, message):
        """Handle a message received from another agent.
        
        Args:
            message (dict): Message to handle
            
        Returns:
            Any: Response to the message
        """
        # Call base class handler first
        response = super().handle_message(message)
        if response is not None:
            return response
        
        # Custom message handling for matching agent
        message_type = message.get('type', 'unknown')
        content = message.get('content', {})
        
        if message_type == 'match_request':
            # Handle match request
            job_id = content.get('job_id')
            candidate_id = content.get('candidate_id')
            
            if job_id and candidate_id:
                return self.match_job_with_candidate(job_id, candidate_id)
            elif job_id:
                return self.match_job_with_all_candidates(job_id)
            elif candidate_id:
                return self.match_candidate_with_all_jobs(candidate_id)
        
        # No custom handling needed, return None
        return None

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
    print("MATCHING AGENT - TEST")
    print("=" * 50)
    
    with MatchingAgent() as agent:
        # Test matching
        print("\nTesting job-candidate matching...")
        
        # Get a job and candidate
        job = agent.db_manager.db.cursor.execute('SELECT id FROM job_descriptions LIMIT 1').fetchone()
        candidate = agent.db_manager.db.cursor.execute('SELECT id FROM candidates LIMIT 1').fetchone()
        
        if job and candidate:
            job_id = job['id']
            candidate_id = candidate['id']
            
            # Match job with candidate
            match_results = agent.match_job_with_candidate(job_id, candidate_id)
            
            if match_results:
                print(f"\nMatch results for Job {job_id} and Candidate {candidate_id}:")
                print(f"Overall match: {match_results['overall_match']:.2f}")
                print(f"Skills match: {match_results['skills_match']:.2f}")
                print(f"Qualifications match: {match_results['qualifications_match']:.2f}")
                print(f"Experience match: {match_results['experience_match']:.2f}")
                print(f"Shortlisted: {match_results['shortlisted']}")
                print("\nNotes:")
                print(match_results['notes'])
        else:
            print("No jobs or candidates found in the database. Please populate the database first.")