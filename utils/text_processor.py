"""
Text processing utilities for the AI-Powered Job Application Screening System.
"""

import logging
import re
import string
from pathlib import Path
import sys
import json
from collections import Counter

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Try importing NLP libraries with fallbacks
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy not installed. Using fallback text processing methods.")
    print("For better results, install spaCy with: pip install spacy")
    print("Then download a model with: python -m spacy download en_core_web_md")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    # Download required NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not installed. Using fallback text processing methods.")
    print("For better results, install NLTK with: pip install nltk")

logger = logging.getLogger(__name__)

class TextProcessor:
    """Text processing utilities for job descriptions and CVs."""
    
    def __init__(self, model_name="en_core_web_md"):
        """Initialize the text processor.
        
        Args:
            model_name (str): Name of the spaCy model to use
        """
        self.nlp = None
        
        # Load spaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            except OSError:
                logger.warning(f"Could not load spaCy model {model_name}. Using fallback methods.")
        
        # Set up NLTK if available
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        else:
            # Simple fallback stopwords
            self.stop_words = set([
                'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                'their', 'theirs', 'themselves', 'this', 'that', 'these', 'those', 'am',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'having', 'do', 'does', 'did', 'doing', 'would', 'should', 'could', 'ought',
                'i\'m', 'you\'re', 'he\'s', 'she\'s', 'it\'s', 'we\'re', 'they\'re',
                'i\'ve', 'you\'ve', 'we\'ve', 'they\'ve', 'i\'d', 'you\'d', 'he\'d',
                'she\'d', 'we\'d', 'they\'d', 'i\'ll', 'you\'ll', 'he\'ll', 'she\'ll',
                'we\'ll', 'they\'ll', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t',
                'haven\'t', 'hadn\'t', 'doesn\'t', 'don\'t', 'didn\'t', 'won\'t', 'wouldn\'t',
                'shan\'t', 'shouldn\'t', 'can\'t', 'cannot', 'couldn\'t', 'mustn\'t', 'let\'s',
                'that\'s', 'who\'s', 'what\'s', 'here\'s', 'there\'s', 'when\'s', 'where\'s',
                'why\'s', 'how\'s', 'to', 'from', 'of', 'with', 'in', 'on', 'by', 'for', 'at',
                'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
            ])
    
    def preprocess_text(self, text):
        """Preprocess text by removing punctuation, lowercasing, and removing stopwords.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            list: Preprocessed tokens
        """
        if self.nlp:
            # Use spaCy for preprocessing
            doc = self.nlp(text)
            tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        elif NLTK_AVAILABLE:
            # Use NLTK for preprocessing
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and token not in string.punctuation]
        else:
            # Fallback preprocessing
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            tokens = text.split()
            tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def extract_key_phrases(self, text, max_phrases=10):
        """Extract key phrases from text.
        
        Args:
            text (str): Text to extract key phrases from
            max_phrases (int): Maximum number of key phrases to extract
            
        Returns:
            list: Extracted key phrases
        """
        if self.nlp:
            # Use spaCy for key phrase extraction
            doc = self.nlp(text)
            
            # Get noun chunks as key phrases
            noun_chunks = list(doc.noun_chunks)
            
            # Get named entities as key phrases
            entities = [ent for ent in doc.ents if ent.label_ in ['SKILL', 'ORG', 'PRODUCT', 'WORK_OF_ART']]
            
            # Combine and sort by length (prefer longer phrases)
            key_phrases = list(set([chunk.text for chunk in noun_chunks] + [ent.text for ent in entities]))
            key_phrases.sort(key=len, reverse=True)
            
            return key_phrases[:max_phrases]
        else:
            # Fallback method: use the most common word sequences
            tokens = self.preprocess_text(text)
            
            # Generate n-grams
            bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
            trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
            
            # Count frequencies
            phrase_counts = Counter(bigrams + trigrams)
            
            # Get the most common phrases
            return [phrase for phrase, _ in phrase_counts.most_common(max_phrases)]
    
    def extract_skills_from_text(self, text):
        """Extract skills from text.
        
        Args:
            text (str): Text to extract skills from
            
        Returns:
            list: Extracted skills
        """
        skills = []
        
        # Common technical skills to search for
        common_skills = [
            "Python", "Java", "C++", "JavaScript", "HTML", "CSS", "SQL", 
            "React", "Angular", "Node.js", "AWS", "Azure", "Docker", "Kubernetes",
            "Machine Learning", "Deep Learning", "Data Analysis", "TensorFlow", "PyTorch",
            "Git", "Linux", "Windows", "Excel", "Word", "PowerPoint", 
            "Project Management", "Agile", "Scrum", "Communication", "Leadership",
            "Problem Solving", "Critical Thinking", "Team Management", "Research",
            "Data Science", "Artificial Intelligence", "Natural Language Processing",
            "Cloud Computing", "DevOps", "UI/UX Design", "Mobile Development",
            "REST APIs", "Microservices", "Database Design", "Statistics"
        ]
        
        # Look for these skills in the text
        for skill in common_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                skills.append(skill)
        
        # Look for skill sections in the text
        skills_section_pattern = r'(?:Skills required|Key skills|Technical skills|Required skills):\s*(.*?)(?:\n\n|\n[A-Z]|\Z)'
        skills_section = re.search(skills_section_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if skills_section:
            skills_text = skills_section.group(1)
            
            # Split by commas, bullets, or newlines
            raw_skill_items = re.split(r'[,•\n]', skills_text)
            
            # Clean and add to skills list
            for item in raw_skill_items:
                item = item.strip()
                if item and len(item) > 2 and item.lower() not in [s.lower() for s in skills]:
                    skills.append(item)
        
        # If using spaCy, look for phrases that might be skills
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract noun phrases that aren't already in the skills list
            for chunk in doc.noun_chunks:
                if 3 < len(chunk.text) < 30 and chunk.text.lower() not in [s.lower() for s in skills]:
                    # Check if the phrase contains skill-related words
                    skill_indicators = ['experience', 'knowledge', 'proficiency', 'skilled', 'expertise']
                    has_indicator = any(indicator in chunk.text.lower() for indicator in skill_indicators)
                    
                    if has_indicator:
                        skills.append(chunk.text)
        
        return list(set(skills))  # Remove duplicates
    
    def extract_qualifications_from_text(self, text):
        """Extract qualifications from text.
        
        Args:
            text (str): Text to extract qualifications from
            
        Returns:
            list: Extracted qualifications
        """
        qualifications = []
        
        # Common qualifications to look for
        degree_patterns = [
            r'(?:Bachelor|B\.S\.|BS|B\.A\.|BA)(?:\'s)? (?:degree|of|in) [A-Za-z\s]+',
            r'(?:Master|M\.S\.|MS|M\.A\.|MA)(?:\'s)? (?:degree|of|in) [A-Za-z\s]+',
            r'(?:Doctor|Ph\.D\.|PhD)(?:\'s)? (?:degree|of|in) [A-Za-z\s]+',
            r'(?:Associate|A\.S\.|AS|A\.A\.|AA)(?:\'s)? (?:degree|of|in) [A-Za-z\s]+'
        ]
        
        # Extract degrees
        for pattern in degree_patterns:
            degree_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in degree_matches:
                if match not in qualifications:
                    qualifications.append(match)
        
        # Look for years of experience
        experience_patterns = [
            r'(\d+\+?\s+years?(?:\s+of)?\s+experience)',
            r'(experience(?:\s+of)?\s+\d+\+?\s+years?)'
        ]
        
        for pattern in experience_patterns:
            exp_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in exp_matches:
                if match not in qualifications:
                    qualifications.append(match)
        
        # Look for certifications
        cert_patterns = [
            r'([A-Za-z]+(?:\s+[A-Za-z]+){0,3} certification)',
            r'(certified\s+[A-Za-z]+(?:\s+[A-Za-z]+){0,3})'
        ]
        
        for pattern in cert_patterns:
            cert_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in cert_matches:
                if match not in qualifications:
                    qualifications.append(match)
        
        # Look for qualifications section
        qual_section_pattern = r'(?:Qualifications|Requirements|Required Qualifications):\s*(.*?)(?:\n\n|\n[A-Z]|\Z)'
        qual_section = re.search(qual_section_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if qual_section:
            qual_text = qual_section.group(1)
            
            # Split by bullets or newlines
            raw_qual_items = re.split(r'[•\n]', qual_text)
            
            # Clean and add to qualifications list
            for item in raw_qual_items:
                item = item.strip()
                if item and len(item) > 10 and item not in qualifications:
                    qualifications.append(item)
        
        return qualifications
    
    def extract_responsibilities_from_text(self, text):
        """Extract responsibilities from text.
        
        Args:
            text (str): Text to extract responsibilities from
            
        Returns:
            list: Extracted responsibilities
        """
        responsibilities = []
        
        # Look for responsibilities section
        resp_section_pattern = r'(?:Responsibilities|Duties|Key Responsibilities|Job Duties):\s*(.*?)(?:\n\n|\n[A-Z]|\Z)'
        resp_section = re.search(resp_section_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if resp_section:
            resp_text = resp_section.group(1)
            
            # Split by bullets or newlines
            raw_resp_items = re.split(r'[•\n]', resp_text)
            
            # Clean and add to responsibilities list
            for item in raw_resp_items:
                item = item.strip()
                if item and len(item) > 10:
                    responsibilities.append(item)
        
        # If using spaCy, look for verb phrases that might be responsibilities
        if self.nlp and not responsibilities:
            doc = self.nlp(text)
            
            # Look for verb phrases
            for token in doc:
                if token.pos_ == "VERB" and not token.is_stop:
                    # Get the subtree as a potential responsibility
                    subtree = [t.text for t in token.subtree]
                    phrase = ' '.join(subtree)
                    
                    # Only add if it's a reasonable length
                    if 10 < len(phrase) < 100:
                        responsibilities.append(phrase)
        
        return responsibilities
    
    def calculate_similarity(self, text1, text2):
        """Calculate the semantic similarity between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if self.nlp:
            # Use spaCy's built-in similarity
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            if not doc1.vector_norm or not doc2.vector_norm:
                # Fall back to token overlap if vectors are empty
                return self._calculate_token_overlap(text1, text2)
            
            return doc1.similarity(doc2)
        else:
            # Fallback: use token overlap
            return self._calculate_token_overlap(text1, text2)
    
    def _calculate_token_overlap(self, text1, text2):
        """Calculate the token overlap between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Overlap score between 0 and 1
        """
        tokens1 = set(self.preprocess_text(text1))
        tokens2 = set(self.preprocess_text(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def calculate_skills_match(self, job_skills, candidate_skills):
        """Calculate the match score between job skills and candidate skills.
        
        Args:
            job_skills (list): Skills required for the job
            candidate_skills (list): Skills possessed by the candidate
            
        Returns:
            float: Skills match score between 0 and 1
        """
        if not job_skills or not candidate_skills:
            return 0.0
        
        # Normalize skills (lowercase for comparison)
        job_skills_norm = [skill.lower() for skill in job_skills]
        candidate_skills_norm = [skill.lower() for skill in candidate_skills]
        
        # Calculate exact matches
        exact_matches = set(job_skills_norm).intersection(set(candidate_skills_norm))
        exact_match_score = len(exact_matches) / len(job_skills_norm)
        
        # If using spaCy, also calculate semantic similarity for non-exact matches
        if self.nlp:
            # For each job skill that wasn't an exact match, find the best semantic match
            remaining_job_skills = [skill for skill in job_skills_norm if skill not in exact_matches]
            
            if not remaining_job_skills:
                return exact_match_score
            
            semantic_match_scores = []
            
            for job_skill in remaining_job_skills:
                best_match_score = 0.0
                
                for candidate_skill in candidate_skills_norm:
                    if candidate_skill in exact_matches:
                        continue  # Skip exact matches already counted
                        
                    similarity = self.calculate_similarity(job_skill, candidate_skill)
                    best_match_score = max(best_match_score, similarity)
                
                semantic_match_scores.append(best_match_score)
            
            # Calculate weighted average of exact and semantic matches
            semantic_match_avg = sum(semantic_match_scores) / len(remaining_job_skills)
            semantic_weight = 0.5  # How much weight to give to semantic matches vs exact matches
            
            return exact_match_score * (1 - semantic_weight) + semantic_match_avg * semantic_weight
        
        # Without spaCy, just return the exact match score
        return exact_match_score
    
    def calculate_qualifications_match(self, job_quals, candidate_quals):
        """Calculate the match score between job qualifications and candidate qualifications.
        
        Args:
            job_quals (list): Qualifications required for the job
            candidate_quals (list): Qualifications possessed by the candidate
            
        Returns:
            float: Qualifications match score between 0 and 1
        """
        if not job_quals or not candidate_quals:
            return 0.0
        
        # Check for education requirements
        job_edu = [qual for qual in job_quals if any(edu in qual.lower() for edu in [
            'degree', 'bachelor', 'master', 'phd', 'doctor', 'diploma', 'certification', 
            'certified', 'license', 'licensed'])]
        
        candidate_edu = [qual for qual in candidate_quals if any(edu in qual.lower() for edu in [
            'degree', 'bachelor', 'master', 'phd', 'doctor', 'diploma', 'certification', 
            'certified', 'license', 'licensed'])]
        
        # If there are education requirements but candidate has none, reduce score
        edu_score = 0.0
        if job_edu and not candidate_edu:
            edu_score = 0.0
        elif not job_edu:
            edu_score = 1.0  # No education requirements
        else:
            # Calculate education match
            edu_match = 0.0
            for job_req in job_edu:
                best_match = 0.0
                for candidate_qual in candidate_edu:
                    match = self.calculate_similarity(job_req, candidate_qual)
                    best_match = max(best_match, match)
                edu_match += best_match
            
            edu_score = edu_match / len(job_edu)
        
        # Check for experience requirements
        job_exp = [qual for qual in job_quals if 'experience' in qual.lower()]
        candidate_exp = [qual for qual in candidate_quals if 'experience' in qual.lower()]
        
        # If there are experience requirements but candidate has none, reduce score
        exp_score = 0.0
        if job_exp and not candidate_exp:
            exp_score = 0.0
        elif not job_exp:
            exp_score = 1.0  # No experience requirements
        else:
            # Calculate experience match
            exp_match = 0.0
            for job_req in job_exp:
                best_match = 0.0
                for candidate_qual in candidate_exp:
                    match = self.calculate_similarity(job_req, candidate_qual)
                    best_match = max(best_match, match)
                exp_match += best_match
            
            exp_score = exp_match / len(job_exp)
        
        # Overall match score is average of education and experience scores
        return (edu_score + exp_score) / 2
    
    def calculate_overall_match(self, job_data, candidate_data):
        """Calculate the overall match score between a job and a candidate.
        
        Args:
            job_data (dict): Job data including skills, qualifications, and responsibilities
            candidate_data (dict): Candidate data including skills, qualifications, and experience
            
        Returns:
            dict: Match results including overall score and component scores
        """
        # Calculate skills match
        skills_match = self.calculate_skills_match(job_data['skills'], candidate_data['skills'])
        
        # Calculate qualifications match
        quals_match = self.calculate_qualifications_match(job_data['qualifications'], candidate_data['qualifications'])
        
        # Calculate experience match based on responsibility-experience similarity
        exp_match = 0.0
        if job_data.get('responsibilities') and candidate_data.get('experience'):
            job_resp_text = ' '.join(job_data['responsibilities'])
            candidate_exp_text = ' '.join(candidate_data['experience'])
            exp_match = self.calculate_similarity(job_resp_text, candidate_exp_text)
        
        # Calculate overall match score with weights
        weights = {
            'skills': 0.5,      # Skills are most important
            'quals': 0.3,       # Qualifications are second
            'exp': 0.2          # Experience is third
        }
        
        overall_match = (
            skills_match * weights['skills'] +
            quals_match * weights['quals'] +
            exp_match * weights['exp']
        )
        
        return {
            'overall_match': overall_match,
            'skills_match': skills_match,
            'qualifications_match': quals_match,
            'experience_match': exp_match
        }
    
    def summarize_job_description(self, title, description):
        """Summarize a job description by extracting skills, qualifications, and responsibilities.
        
        Args:
            title (str): Job title
            description (str): Job description text
            
        Returns:
            dict: Summarized job data
        """
        # Extract skills, qualifications, and responsibilities
        skills = self.extract_skills_from_text(description)
        qualifications = self.extract_qualifications_from_text(description)
        responsibilities = self.extract_responsibilities_from_text(description)
        
        # Create the summary
        summary = {
            'title': title,
            'description': description,
            'skills': skills,
            'qualifications': qualifications,
            'responsibilities': responsibilities
        }
        
        return summary 