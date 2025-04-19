"""
Database package for the AI-Powered Job Application Screening System.

This package contains the following modules:
- models: SQLite database models
- db_manager: Database interaction functions
"""

from database.models import Database
from database.db_manager import DBManager

__all__ = [
    'Database',
    'DBManager'
] 