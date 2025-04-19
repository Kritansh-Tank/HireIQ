"""
Utility for extracting text from PDF files.
"""

import logging
import os
import re
from pathlib import Path
import sys
import io
import traceback

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Try to import PyPDF2, and provide helpful error message if it's not installed
try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not installed. Please install it with: pip install PyPDF2")
    sys.exit(1)

logger = logging.getLogger(__name__)

class PDFExtractor:
    """A utility class for extracting text from PDF files."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            logger.info(f"Attempting to extract text from PDF: {pdf_path}")
            
            # Check if the file exists and get its size
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file does not exist: {pdf_path}")
                return ""
                
            file_size = os.path.getsize(pdf_path)
            logger.info(f"PDF file size: {file_size} bytes")
            
            # Special handling for small PDF files that might be corrupted or malformed
            if file_size < 100:
                logger.warning(f"PDF file is very small ({file_size} bytes), it might be corrupted: {pdf_path}")
                return f"[Could not extract text from small PDF file: {os.path.basename(pdf_path)}]"
            
            # Open the file in binary mode to avoid encoding issues
            with open(pdf_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ''
                    
                    # Extract text from each page
                    num_pages = len(pdf_reader.pages)
                    logger.info(f"PDF has {num_pages} pages")
                    
                    for page_num in range(num_pages):
                        try:
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            text += page_text + '\n'
                        except Exception as e:
                            logger.error(f"Error extracting text from page {page_num}: {str(e)}")
                            text += f"[Error extracting page {page_num}]\n"
                    
                    # Clean up the text
                    text = PDFExtractor._clean_text(text)
                    
                    # Log a sample of the extracted text for debugging
                    sample = text[:100] + "..." if len(text) > 100 else text
                    logger.info(f"Extracted text sample: {sample}")
                    
                    return text
                    
                except PyPDF2.errors.PdfReadError as e:
                    logger.error(f"PyPDF2 error reading PDF {pdf_path}: {str(e)}")
                    # Try fallback method if available
                    return f"[Error: Could not read PDF file: {str(e)}]"
                    
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return f"[Error: {str(e)}]"
    
    @staticmethod
    def extract_text_from_pdf_bytes(pdf_bytes):
        """Extract text from PDF bytes.
        
        Args:
            pdf_bytes (bytes): PDF content as bytes
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            with io.BytesIO(pdf_bytes) as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                
                # Extract text from each page
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + '\n'
                
                # Clean up the text
                text = PDFExtractor._clean_text(text)
                return text
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF bytes: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
    
    @staticmethod
    def _clean_text(text):
        """Clean up the extracted text.
        
        Args:
            text (str): The text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove multiple consecutive spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    @staticmethod
    def extract_basic_info(text):
        """Extract basic information from CV text.
        
        Args:
            text (str): The CV text
            
        Returns:
            dict: Extracted information (name, email, phone)
        """
        # Initialize result
        info = {
            'name': None,
            'email': None,
            'phone': None
        }
        
        # If text is empty or contains an error message, return empty info
        if not text or text.startswith('[Error:'):
            return info
        
        # Extract email (simple pattern)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        if email_matches:
            info['email'] = email_matches[0]
        
        # Extract phone number (simple pattern, can be expanded)
        phone_pattern = r'\b(\+\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b'
        phone_matches = re.findall(phone_pattern, text)
        if phone_matches:
            info['phone'] = phone_matches[0]
        
        # Extract name (heuristic - first line or after "Name:")
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and 'resume' not in line.lower() and 'cv' not in line.lower():
                name_pattern = r'^([A-Z][a-z]+ [A-Z][a-z]+)'
                name_match = re.search(name_pattern, line)
                if name_match:
                    info['name'] = name_match.group(1)
                    break
            
            # Look for "Name:" pattern
            name_label_pattern = r'Name:?\s*([A-Z][a-z]+ [A-Z][a-z]+)'
            name_label_match = re.search(name_label_pattern, line)
            if name_label_match:
                info['name'] = name_label_match.group(1)
                break
        
        return info
    
    @staticmethod
    def extract_skills(text):
        """Extract skills from CV text.
        
        Args:
            text (str): The CV text
            
        Returns:
            list: Extracted skills
        """
        skills = []
        
        # Look for skills section
        skills_section_pattern = r'(?:SKILLS|TECHNICAL SKILLS|PROFESSIONAL SKILLS).*?\n(.*?)(?:\n\n|\n[A-Z\s]+:|$)'
        skills_section = re.search(skills_section_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if skills_section:
            # Extract skills from the skills section
            skills_text = skills_section.group(1)
            
            # Split by commas, bullets, or newlines
            raw_skills = re.split(r'[,•\n]', skills_text)
            
            # Clean and add to skills list
            for skill in raw_skills:
                skill = skill.strip()
                if skill and len(skill) > 2:  # Avoid short or empty strings
                    skills.append(skill)
        
        # Common technical skills to look for
        common_skills = [
            "Python", "Java", "C++", "JavaScript", "HTML", "CSS", "SQL", 
            "React", "Angular", "Node.js", "AWS", "Azure", "Docker", "Kubernetes",
            "Machine Learning", "Deep Learning", "Data Analysis", "TensorFlow", "PyTorch",
            "Git", "Linux", "Windows", "Excel", "Word", "PowerPoint", 
            "Project Management", "Agile", "Scrum", "Communication", "Leadership"
        ]
        
        # Look for common skills directly in the text
        for skill in common_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                if skill not in skills:
                    skills.append(skill)
        
        return list(set(skills))  # Remove duplicates
    
    @staticmethod
    def extract_education(text):
        """Extract education information from CV text.
        
        Args:
            text (str): The CV text
            
        Returns:
            list: Extracted education information
        """
        education = []
        
        # Look for education section
        education_section_pattern = r'(?:EDUCATION|EDUCATIONAL BACKGROUND|ACADEMIC BACKGROUND).*?\n(.*?)(?:\n\n|\n[A-Z\s]+:|$)'
        education_section = re.search(education_section_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if education_section:
            # Extract education items
            education_text = education_section.group(1)
            
            # Split by newlines or bullets to get education items
            edu_items = re.split(r'\n+|•', education_text)
            
            # Clean and add to education list
            for item in edu_items:
                item = item.strip()
                if item and len(item) > 10:  # Avoid short or empty strings
                    education.append(item)
        
        # Look for degrees directly
        degree_patterns = [
            r'(?:Bachelor|B\.S\.|BS|B\.A\.|BA)(?:\'s)? (?:of|in) [A-Za-z\s]+',
            r'(?:Master|M\.S\.|MS|M\.A\.|MA)(?:\'s)? (?:of|in) [A-Za-z\s]+',
            r'(?:Doctor|Ph\.D\.|PhD)(?:\'s)? (?:of|in) [A-Za-z\s]+'
        ]
        
        for pattern in degree_patterns:
            degree_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in degree_matches:
                if match not in education:
                    education.append(match)
        
        return education
    
    @staticmethod
    def extract_experience(text):
        """Extract work experience from CV text.
        
        Args:
            text (str): The CV text
            
        Returns:
            list: Extracted work experience
        """
        experience = []
        
        # Look for experience section using traditional section headers
        exp_section_pattern = r'(?:EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT).*?\n(.*?)(?:\n\n|\n[A-Z\s]+:|$)'
        exp_section = re.search(exp_section_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if exp_section:
            # Extract experience items
            exp_text = exp_section.group(1)
            
            # Split by double newlines or company patterns to get experience items
            exp_items = re.split(r'\n\n+|(?:\n[A-Z][a-z]+[^a-z\n]+)', exp_text)
            
            # Clean and add to experience list
            for item in exp_items:
                item = item.strip()
                if item and len(item) > 15:  # Avoid short or empty strings
                    experience.append(item)
        
        # If no experience found with the traditional pattern, try alternative approaches
        if not experience:
            # Try to find experience mentioned anywhere in the text
            exp_sentences = []
            
            # Look for sentences containing the word "experience"
            for sentence in re.split(r'(?<=[.!?])\s+', text):
                if re.search(r'\bexperience\b', sentence, re.IGNORECASE):
                    # Clean up the sentence
                    clean_sentence = sentence.strip()
                    if clean_sentence and len(clean_sentence) > 15:  # Avoid short sentences
                        exp_sentences.append(clean_sentence)
            
            # If we found sentences with "experience", add them
            if exp_sentences:
                experience.extend(exp_sentences)
                
            # Try to find paragraphs after "experience" keyword
            if not experience:
                exp_index = text.lower().find("experience")
                if exp_index != -1:
                    # Extract text after the experience keyword
                    following_text = text[exp_index:exp_index+500]  # Get a reasonable chunk
                    # Clean and split by periods or line breaks
                    exp_chunks = re.split(r'(?<=[.!?])\s+|\n+', following_text)
                    
                    # Add non-empty chunks
                    for chunk in exp_chunks:
                        chunk = chunk.strip()
                        if chunk and len(chunk) > 15 and chunk not in experience:
                            experience.append(chunk)
        
        # If still no experience found, look for common experience patterns
        if not experience:
            company_patterns = [
                r'([A-Z][a-zA-Z]+ (?:Inc|LLC|Ltd|Corporation|Corp\.))',
                r'(?:at|for) ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){0,3}) (?:from|between)',
                r'([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){0,3}) \((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
            ]
            
            for pattern in company_patterns:
                company_matches = re.findall(pattern, text)
                for match in company_matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    line = f"Worked at {match}"
                    if line not in experience:
                        experience.append(line)
        
        # If still no experience, check for project work as it might indicate experience
        if not experience and "project" in text.lower():
            project_pattern = r'(?:PROJECT|PROJECTS).*?\n(.*?)(?:\n\n|\n[A-Z\s]+:|$)'
            project_section = re.search(project_pattern, text, re.IGNORECASE | re.DOTALL)
            
            if project_section:
                project_text = project_section.group(1)
                # Split and process project items
                project_items = re.split(r'\n\n+|(?:\n[A-Z][a-z]+[^a-z\n]+)', project_text)
                
                for item in project_items:
                    item = item.strip()
                    if item and len(item) > 15:
                        experience.append(f"Project Experience: {item}")
        
        # Last resort: Extract context around the word "experience" if found but not properly formatted
        if not experience and "experience" in text.lower():
            # Get a window of text around "experience" mention
            exp_match = re.search(r'.{0,50}experience.{0,100}', text, re.IGNORECASE)
            if exp_match:
                context = exp_match.group(0).strip()
                if context and len(context) > 15:
                    experience.append(context)
        
        return experience
    
    @staticmethod
    def extract_cv_data(pdf_path):
        """Extract all relevant data from a CV PDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Extracted CV data
        """
        # Extract CV ID from filename
        cv_id = os.path.basename(pdf_path).split('.')[0]
        logger.info(f"Processing CV with ID: {cv_id}")
        
        # Extract text from PDF
        text = PDFExtractor.extract_text_from_pdf(pdf_path)
        
        # If text extraction failed, provide placeholder data
        if not text or text.startswith('[Error:'):
            logger.warning(f"Text extraction failed for CV: {cv_id}. Using placeholder data.")
            return {
                'cv_id': cv_id,
                'name': cv_id,  # Use CV ID as name
                'email': f"{cv_id}@example.com",
                'phone': "000-000-0000",
                'skills': ["Unable to extract skills"],
                'qualifications': ["Unable to extract qualifications"],
                'experience': ["Unable to extract experience"],
                'cv_text': text or f"[Failed to extract text from {cv_id}.pdf]"
            }
        
        # Extract information
        basic_info = PDFExtractor.extract_basic_info(text)
        skills = PDFExtractor.extract_skills(text)
        education = PDFExtractor.extract_education(text)
        experience = PDFExtractor.extract_experience(text)
        
        # Combine all data
        cv_data = {
            'cv_id': cv_id,
            'name': basic_info['name'] or cv_id,  # Use CV ID if name not found
            'email': basic_info['email'] or f"{cv_id}@example.com",  # Use placeholder if email not found
            'phone': basic_info['phone'] or "000-000-0000",  # Use placeholder if phone not found
            'skills': skills,
            'qualifications': education,
            'experience': experience,
            'cv_text': text
        }
        
        logger.info(f"Successfully extracted data from CV: {cv_id}")
        return cv_data 