"""
Database models for the AI-Powered Job Application Screening System.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

class Database:
    def __init__(self, db_path):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_tables()
        
    def _create_tables(self):
        """Create database tables if they don't exist."""
        # Job Description table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            skills TEXT NOT NULL,  -- JSON array of skills
            qualifications TEXT NOT NULL,  -- JSON array of qualifications
            responsibilities TEXT NOT NULL,  -- JSON array of responsibilities
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Candidate table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cv_id TEXT UNIQUE NOT NULL,  -- CV file identifier (e.g., C3014)
            name TEXT,
            email TEXT,
            phone TEXT,
            skills TEXT NOT NULL,  -- JSON array of skills
            qualifications TEXT NOT NULL,  -- JSON array of qualifications
            experience TEXT NOT NULL,  -- JSON array of experience
            cv_text TEXT NOT NULL,  -- Full text of CV
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Match Results table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            candidate_id INTEGER NOT NULL,
            match_score REAL NOT NULL,  -- Float between 0 and 1
            skills_match_score REAL NOT NULL,
            qualifications_match_score REAL NOT NULL,
            experience_match_score REAL NOT NULL,
            shortlisted BOOLEAN NOT NULL DEFAULT 0,  -- 0=False, 1=True
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES job_descriptions (id),
            FOREIGN KEY (candidate_id) REFERENCES candidates (id)
        )
        ''')
        
        # Interview Requests table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS interview_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            status TEXT NOT NULL,  -- "scheduled", "pending", "canceled", "completed"
            scheduled_date TEXT,
            interview_type TEXT,  -- "phone", "video", "in-person"
            email_sent BOOLEAN NOT NULL DEFAULT 0,
            email_sent_at TIMESTAMP,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (match_id) REFERENCES match_results (id)
        )
        ''')
        
        self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    # Job Description methods
    def add_job_description(self, title, description, skills, qualifications, responsibilities):
        """Add a new job description to the database."""
        skills_json = json.dumps(skills)
        qualifications_json = json.dumps(qualifications)
        responsibilities_json = json.dumps(responsibilities)
        
        self.cursor.execute('''
        INSERT INTO job_descriptions (title, description, skills, qualifications, responsibilities)
        VALUES (?, ?, ?, ?, ?)
        ''', (title, description, skills_json, qualifications_json, responsibilities_json))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_job_description(self, job_id):
        """Get job description by ID."""
        self.cursor.execute('SELECT * FROM job_descriptions WHERE id = ?', (job_id,))
        job = self.cursor.fetchone()
        if job:
            job_dict = dict(job)
            job_dict['skills'] = json.loads(job_dict['skills'])
            job_dict['qualifications'] = json.loads(job_dict['qualifications'])
            job_dict['responsibilities'] = json.loads(job_dict['responsibilities'])
            return job_dict
        return None
        
    def get_all_job_descriptions(self):
        """Get all job descriptions."""
        self.cursor.execute('SELECT * FROM job_descriptions')
        jobs = self.cursor.fetchall()
        results = []
        for job in jobs:
            job_dict = dict(job)
            job_dict['skills'] = json.loads(job_dict['skills'])
            job_dict['qualifications'] = json.loads(job_dict['qualifications'])
            job_dict['responsibilities'] = json.loads(job_dict['responsibilities'])
            results.append(job_dict)
        return results
    
    # Candidate methods
    def add_candidate(self, cv_id, name, email, phone, skills, qualifications, experience, cv_text):
        """Add a new candidate to the database."""
        skills_json = json.dumps(skills)
        qualifications_json = json.dumps(qualifications)
        experience_json = json.dumps(experience)
        
        self.cursor.execute('''
        INSERT INTO candidates (cv_id, name, email, phone, skills, qualifications, experience, cv_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (cv_id, name, email, phone, skills_json, qualifications_json, experience_json, cv_text))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_candidate(self, candidate_id):
        """Get candidate by ID."""
        self.cursor.execute('SELECT * FROM candidates WHERE id = ?', (candidate_id,))
        candidate = self.cursor.fetchone()
        if candidate:
            candidate_dict = dict(candidate)
            candidate_dict['skills'] = json.loads(candidate_dict['skills'])
            candidate_dict['qualifications'] = json.loads(candidate_dict['qualifications'])
            candidate_dict['experience'] = json.loads(candidate_dict['experience'])
            return candidate_dict
        return None
    
    def get_candidate_by_cv_id(self, cv_id):
        """Get candidate by CV ID."""
        self.cursor.execute('SELECT * FROM candidates WHERE cv_id = ?', (cv_id,))
        candidate = self.cursor.fetchone()
        if candidate:
            candidate_dict = dict(candidate)
            candidate_dict['skills'] = json.loads(candidate_dict['skills'])
            candidate_dict['qualifications'] = json.loads(candidate_dict['qualifications'])
            candidate_dict['experience'] = json.loads(candidate_dict['experience'])
            return candidate_dict
        return None
    
    def update_candidate(self, candidate_id, name, email, phone, skills, qualifications, experience, cv_text):
        """Update an existing candidate in the database.
        
        Args:
            candidate_id (int): ID of the candidate to update
            name (str): Candidate name
            email (str): Candidate email
            phone (str): Candidate phone number
            skills (list): List of skills
            qualifications (list): List of qualifications
            experience (list): List of experience items
            cv_text (str): Full text of CV
            
        Returns:
            bool: True if update was successful
        """
        skills_json = json.dumps(skills)
        qualifications_json = json.dumps(qualifications)
        experience_json = json.dumps(experience)
        
        try:
            self.cursor.execute('''
            UPDATE candidates 
            SET name = ?, email = ?, phone = ?, skills = ?, qualifications = ?, experience = ?, cv_text = ? 
            WHERE id = ?
            ''', (name, email, phone, skills_json, qualifications_json, experience_json, cv_text, candidate_id))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating candidate: {str(e)}")
            return False
    
    # Match Results methods
    def add_match_result(self, job_id, candidate_id, match_score, skills_match, qualifications_match, 
                          experience_match, shortlisted=False, notes=None):
        """Add a match result to the database."""
        self.cursor.execute('''
        INSERT INTO match_results (job_id, candidate_id, match_score, skills_match_score, 
                                   qualifications_match_score, experience_match_score, shortlisted, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (job_id, candidate_id, match_score, skills_match, qualifications_match, 
              experience_match, 1 if shortlisted else 0, notes))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_match_results(self, job_id, shortlisted_only=False):
        """Get all match results for a job, optionally filtered to shortlisted candidates only."""
        query = 'SELECT * FROM match_results WHERE job_id = ?'
        params = [job_id]
        
        if shortlisted_only:
            query += ' AND shortlisted = 1'
            
        self.cursor.execute(query, params)
        results = [dict(row) for row in self.cursor.fetchall()]
        return results
    
    def update_shortlist_status(self, match_id, shortlisted):
        """Update the shortlisted status of a match."""
        self.cursor.execute('''
        UPDATE match_results SET shortlisted = ? WHERE id = ?
        ''', (1 if shortlisted else 0, match_id))
        self.conn.commit()
        return self.cursor.rowcount > 0
    
    # Interview Request methods
    def add_interview_request(self, match_id, status="pending", scheduled_date=None, 
                              interview_type="video", notes=None):
        """Add an interview request."""
        self.cursor.execute('''
        INSERT INTO interview_requests (match_id, status, scheduled_date, interview_type, notes)
        VALUES (?, ?, ?, ?, ?)
        ''', (match_id, status, scheduled_date, interview_type, notes))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def update_interview_email_sent(self, interview_id):
        """Mark an interview request as having had the email sent."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute('''
        UPDATE interview_requests SET email_sent = 1, email_sent_at = ? WHERE id = ?
        ''', (now, interview_id))
        self.conn.commit()
        return self.cursor.rowcount > 0
    
    def get_pending_interviews(self):
        """Get all pending interview requests that need emails sent."""
        self.cursor.execute('''
        SELECT ir.id, ir.match_id, ir.status, ir.scheduled_date, ir.interview_type, ir.notes,
               m.job_id, m.candidate_id, j.title as job_title, c.name as candidate_name, 
               c.email as candidate_email
        FROM interview_requests ir
        JOIN match_results m ON ir.match_id = m.id
        JOIN job_descriptions j ON m.job_id = j.id
        JOIN candidates c ON m.candidate_id = c.id
        WHERE ir.email_sent = 0
        ''')
        return [dict(row) for row in self.cursor.fetchall()]

    def reset_database(self):
        """Clear all data from the database tables.
        
        Returns:
            bool: True if reset was successful, False otherwise
        """
        try:
            # Clear all tables in reverse order of dependencies
            self.cursor.execute("DELETE FROM interview_requests")
            self.cursor.execute("DELETE FROM match_results")
            self.cursor.execute("DELETE FROM candidates")
            self.cursor.execute("DELETE FROM job_descriptions")
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            return False 