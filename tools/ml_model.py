"""
ML Model Tool

This module provides machine learning model functionality for the agent system.
"""

import logging
import numpy as np
import os
import pickle
import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config
from utils.embeddings import EmbeddingUtility

logger = logging.getLogger(__name__)

class MLModelTool:
    """Machine learning model tool for job candidate matching and other ML tasks."""
    
    def __init__(self, model_dir=None):
        """Initialize the ML model tool.
        
        Args:
            model_dir (str, optional): Directory to store ML models
        """
        self.model_dir = model_dir or config.ML_MODELS_DIR
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize embedding utility
        self.embedding_util = EmbeddingUtility()
        
        # Initialize models
        self.tfidf_vectorizer = None
        self.skill_classifier = None
        
        # Load models if they exist
        self._load_models()
        
        logger.info("Initialized ML model tool")
    
    def _load_models(self):
        """Load ML models from disk."""
        # TF-IDF vectorizer
        tfidf_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
        if os.path.exists(tfidf_path):
            try:
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                    logger.info("Loaded TF-IDF vectorizer from disk")
            except Exception as e:
                logger.error(f"Error loading TF-IDF vectorizer: {str(e)}")
                self.tfidf_vectorizer = None
    
    def _save_models(self):
        """Save ML models to disk."""
        # TF-IDF vectorizer
        if self.tfidf_vectorizer:
            tfidf_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
            try:
                with open(tfidf_path, 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f)
                    logger.info("Saved TF-IDF vectorizer to disk")
            except Exception as e:
                logger.error(f"Error saving TF-IDF vectorizer: {str(e)}")
    
    def train_tfidf_vectorizer(self, documents):
        """Train a TF-IDF vectorizer on a set of documents.
        
        Args:
            documents (list): List of text documents
            
        Returns:
            bool: True if training was successful
        """
        try:
            logger.info("Training TF-IDF vectorizer")
            
            # Create and train vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.tfidf_vectorizer.fit(documents)
            
            # Save models
            self._save_models()
            
            logger.info(f"Trained TF-IDF vectorizer with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error training TF-IDF vectorizer: {str(e)}")
            return False
    
    def match_job_candidates(self, job_description, candidate_descriptions, threshold=0.6, top_k=5):
        """Match job with candidates using ML models.
        
        This method uses multiple techniques:
        1. TF-IDF vectorization + cosine similarity (if trained)
        2. Semantic embedding similarity
        
        Args:
            job_description (str): Job description text
            candidate_descriptions (list): List of candidate description texts
            threshold (float, optional): Minimum match score threshold
            top_k (int, optional): Maximum number of top matches to return
            
        Returns:
            list: List of (index, score, candidate) tuples in descending order of match score
        """
        logger.info(f"Matching job with {len(candidate_descriptions)} candidates")
        
        # Results from each method
        results = []
        
        # Method 1: TF-IDF vectorization + cosine similarity
        if self.tfidf_vectorizer:
            try:
                # Transform job description
                job_vector = self.tfidf_vectorizer.transform([job_description])
                
                # Transform candidate descriptions
                candidate_vectors = self.tfidf_vectorizer.transform(candidate_descriptions)
                
                # Compute similarities
                similarities = cosine_similarity(job_vector, candidate_vectors)[0]
                
                # Add to results
                for i, similarity in enumerate(similarities):
                    if similarity >= threshold:
                        results.append((i, float(similarity), candidate_descriptions[i]))
                
                logger.debug(f"TF-IDF matching found {len(results)} candidates above threshold")
                
            except Exception as e:
                logger.error(f"Error in TF-IDF matching: {str(e)}")
        
        # Method 2: Semantic embedding similarity
        try:
            # Get job embedding
            job_embedding = self.embedding_util.get_embedding(job_description)
            
            # Get candidate embeddings
            for i, candidate_desc in enumerate(candidate_descriptions):
                # Get candidate embedding
                candidate_embedding = self.embedding_util.get_embedding(candidate_desc)
                
                if job_embedding is not None and candidate_embedding is not None:
                    # Compute cosine similarity
                    similarity = float(np.dot(job_embedding, candidate_embedding) / 
                                      (np.linalg.norm(job_embedding) * np.linalg.norm(candidate_embedding)))
                    
                    # Check if candidate already in results
                    existing_index = next((idx for idx, r in enumerate(results) if r[0] == i), None)
                    
                    if existing_index is not None:
                        # Update score to average of both methods
                        old_score = results[existing_index][1]
                        results[existing_index] = (i, (old_score + similarity) / 2, candidate_descriptions[i])
                    elif similarity >= threshold:
                        # Add new result
                        results.append((i, similarity, candidate_descriptions[i]))
            
            logger.debug(f"Semantic matching found {len(results)} candidates above threshold")
            
        except Exception as e:
            logger.error(f"Error in semantic matching: {str(e)}")
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        return results[:top_k]
    
    def extract_skills(self, text):
        """Extract skills from text using ML models.
        
        This method uses Ollama to extract skills from a text.
        
        Args:
            text (str): Text to extract skills from
            
        Returns:
            list: List of extracted skills
        """
        from utils.ollama_client import OllamaClient
        
        # Create a temporary client with the default model
        client = OllamaClient()
        
        try:
            # Prompt for skill extraction
            prompt = f"""Extract a list of professional skills from the following text. 
            Return only the skills as a comma-separated list with no additional text or explanations.
            
            Text: {text}
            
            Skills:"""
            
            # Generate skills
            response = client.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1
            )
            
            # Process response
            if response:
                # Split by commas and clean up
                skills = [skill.strip() for skill in response.split(',')]
                
                # Filter out empty skills
                skills = [skill for skill in skills if skill]
                
                return skills
            else:
                return []
            
        except Exception as e:
            logger.error(f"Error extracting skills: {str(e)}")
            return []
        finally:
            client.close()
    
    def analyze_job_requirements(self, job_description):
        """Analyze job requirements from a job description.
        
        Args:
            job_description (str): Job description text
            
        Returns:
            dict: Dictionary of job requirements analysis
        """
        from utils.ollama_client import OllamaClient
        
        # Create a temporary client with the default model
        client = OllamaClient()
        
        try:
            # Prompt for job requirements analysis
            prompt = f"""Analyze the following job description and extract:
            1. Required skills (comma-separated list)
            2. Preferred skills (comma-separated list)
            3. Required experience in years
            4. Required education level
            5. Job level (entry, mid, senior, executive)
            
            Format the response as JSON with these keys: required_skills, preferred_skills, experience_years, education, job_level.
            
            Job description: {job_description}"""
            
            # Generate analysis
            response = client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse JSON response
            try:
                # Extract JSON from response (in case there's other text)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    analysis = json.loads(json_str)
                    
                    # Ensure all keys exist
                    if not all(k in analysis for k in ['required_skills', 'preferred_skills', 'experience_years', 'education', 'job_level']):
                        logger.warning("Job analysis missing some keys, filling in defaults")
                        
                        # Add missing keys with defaults
                        analysis.setdefault('required_skills', [])
                        analysis.setdefault('preferred_skills', [])
                        analysis.setdefault('experience_years', 0)
                        analysis.setdefault('education', "Not specified")
                        analysis.setdefault('job_level', "Not specified")
                    
                    return analysis
                else:
                    logger.warning("Could not extract JSON from job analysis response")
                    return {
                        'required_skills': [],
                        'preferred_skills': [],
                        'experience_years': 0,
                        'education': "Not specified",
                        'job_level': "Not specified"
                    }
            except json.JSONDecodeError:
                logger.error("Error parsing job analysis JSON")
                return {
                    'required_skills': [],
                    'preferred_skills': [],
                    'experience_years': 0,
                    'education': "Not specified",
                    'job_level': "Not specified"
                }
            
        except Exception as e:
            logger.error(f"Error analyzing job requirements: {str(e)}")
            return {
                'required_skills': [],
                'preferred_skills': [],
                'experience_years': 0,
                'education': "Not specified",
                'job_level': "Not specified"
            }
        finally:
            client.close()
    
    def close(self):
        """Close the ML model tool and free resources."""
        logger.info("Closing ML model tool")

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
    
    ml_tool = MLModelTool()
    
    # Test skill extraction
    resume_text = """
    John Doe
    Software Engineer
    
    Skills: Python, JavaScript, React, Docker, AWS, Machine Learning
    
    Experience:
    - Developed web applications using React and Node.js
    - Implemented machine learning models using scikit-learn and TensorFlow
    - Deployed applications to AWS using Docker containers
    
    Education:
    BSc in Computer Science, 2018
    """
    
    skills = ml_tool.extract_skills(resume_text)
    print("\nExtracted skills:")
    print(skills)
    
    # Test job requirements analysis
    job_description = """
    Senior Software Engineer - Machine Learning
    
    We are looking for a Senior Software Engineer specializing in Machine Learning to join our team.
    
    Requirements:
    - 5+ years of software development experience
    - Strong Python programming skills
    - Experience with machine learning frameworks (TensorFlow, PyTorch)
    - Bachelor's degree in Computer Science or related field
    - Experience with cloud platforms (AWS, GCP)
    
    Preferred:
    - Master's degree in Machine Learning or AI
    - Experience with NLP and computer vision
    - Knowledge of Docker and Kubernetes
    """
    
    analysis = ml_tool.analyze_job_requirements(job_description)
    print("\nJob requirements analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Test candidate matching
    candidates = [
        """Experienced software engineer with 6 years of Python development. Expertise in machine learning using TensorFlow and PyTorch. Bachelor's in Computer Science with cloud platform experience on AWS.""",
        """Frontend developer with 3 years of React experience. Some Python knowledge but no machine learning background. Associate's degree in Web Development.""",
        """Machine learning engineer with 4 years of experience. Expert in TensorFlow, PyTorch, and NLP. Master's degree in AI. Limited cloud experience."""
    ]
    
    matches = ml_tool.match_job_candidates(job_description, candidates)
    print("\nCandidate matches:")
    for i, score, candidate in matches:
        print(f"Candidate {i+1}: {score:.4f}")
    
    ml_tool.close() 