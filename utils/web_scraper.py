"""
Web Scraper Utility

This module provides a web scraper that can fetch and process information from websites.
"""

import logging
import requests
import os
import time
import json
import hashlib
from bs4 import BeautifulSoup
from pathlib import Path
import sys
from urllib.parse import urlparse, urljoin
import traceback

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config

logger = logging.getLogger(__name__)

class WebScraper:
    """Web scraper utility for fetching and processing information from websites."""
    
    def __init__(self, cache_dir=None, user_agent=None, timeout=None):
        """Initialize the web scraper.
        
        Args:
            cache_dir (str, optional): Directory to cache scraped content
            user_agent (str, optional): User agent to use for requests
            timeout (int, optional): Timeout in seconds for requests
        """
        self.cache_dir = cache_dir or config.SCRAPER_CACHE_DIR
        self.user_agent = user_agent or config.SCRAPER_USER_AGENT
        self.timeout = timeout or config.SCRAPER_TIMEOUT
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def _get_cache_path(self, url):
        """Get the cache path for a URL.
        
        Args:
            url (str): URL to get cache path for
            
        Returns:
            str: Cache path
        """
        # Create a hash of the URL to use as the filename
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.json")
    
    def fetch_url(self, url, use_cache=True, force_refresh=False):
        """Fetch content from a URL.
        
        Args:
            url (str): URL to fetch
            use_cache (bool, optional): Whether to use cached content
            force_refresh (bool, optional): Whether to force refresh the cache
            
        Returns:
            dict: Dictionary with URL data including status, content, and metadata
        """
        # Check cache first if enabled and not forcing refresh
        if use_cache and not force_refresh:
            cache_path = self._get_cache_path(url)
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        logger.debug(f"Loaded URL from cache: {url}")
                        return cached_data
                except Exception as e:
                    logger.warning(f"Error loading cached URL: {str(e)}")
        
        # Fetch from web if not cached or cache loading failed or force refresh
        try:
            logger.info(f"Fetching URL: {url}")
            start_time = time.time()
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            
            # Process response based on content type
            if 'text/html' in content_type:
                # Process as HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract useful data
                title = soup.title.string if soup.title else ""
                
                # Extract main text content
                main_content = self._extract_main_content(soup)
                
                # Extract links
                links = self._extract_links(soup, url)
                
                # Create structured result
                result = {
                    'url': url,
                    'status': response.status_code,
                    'content_type': 'html',
                    'title': title,
                    'text': main_content,
                    'links': links,
                    'timestamp': time.time(),
                    'headers': dict(response.headers)
                }
            elif 'application/json' in content_type:
                # Process as JSON
                json_data = response.json()
                
                # Create structured result
                result = {
                    'url': url,
                    'status': response.status_code,
                    'content_type': 'json',
                    'data': json_data,
                    'timestamp': time.time(),
                    'headers': dict(response.headers)
                }
            else:
                # Process as raw text
                result = {
                    'url': url,
                    'status': response.status_code,
                    'content_type': 'text',
                    'text': response.text[:10000],  # Limit text to avoid huge files
                    'timestamp': time.time(),
                    'headers': dict(response.headers)
                }
            
            end_time = time.time()
            logger.debug(f"Fetched URL in {end_time - start_time:.2f} seconds: {url}")
            
            # Cache the result if enabled
            if use_cache:
                cache_path = self._get_cache_path(url)
                
                try:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                        logger.debug(f"Cached URL: {url}")
                except Exception as e:
                    logger.warning(f"Error caching URL: {str(e)}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            
            # Create error result
            error_result = {
                'url': url,
                'status': -1,
                'content_type': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
            
            return error_result
    
    def _extract_main_content(self, soup):
        """Extract the main content from an HTML page.
        
        Args:
            soup (BeautifulSoup): BeautifulSoup object for the page
            
        Returns:
            str: Main content text
        """
        # Try to find main content elements
        main_elements = soup.find_all(['article', 'main', 'div'], class_=['content', 'main', 'article', 'post'])
        
        if main_elements:
            # Use the largest main element as the main content
            main_element = max(main_elements, key=lambda x: len(x.get_text()))
            
            # Extract text
            text = main_element.get_text(separator='\n', strip=True)
        else:
            # Use the body as the main content
            body = soup.find('body')
            
            if body:
                # Remove script and style elements
                for script in body(['script', 'style']):
                    script.decompose()
                
                # Extract text
                text = body.get_text(separator='\n', strip=True)
            else:
                # Use the entire document
                text = soup.get_text(separator='\n', strip=True)
        
        # Clean up text
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        
        return '\n'.join(lines)
    
    def _extract_links(self, soup, base_url):
        """Extract links from an HTML page.
        
        Args:
            soup (BeautifulSoup): BeautifulSoup object for the page
            base_url (str): Base URL for resolving relative links
            
        Returns:
            list: List of links
        """
        links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            
            # Skip empty or javascript links
            if not href or href.startswith('javascript:'):
                continue
            
            # Resolve relative links
            absolute_url = urljoin(base_url, href)
            
            # Extract link text
            link_text = a.get_text(strip=True)
            
            links.append({
                'url': absolute_url,
                'text': link_text
            })
        
        return links
    
    def search_with_extraction(self, query, num_results=5, site=None):
        """Search the web for information and extract content from pages.
        
        This is a placeholder method. In a real implementation, this would
        use a search engine API like Google or Bing to search for information,
        then extract content from the top results.
        
        Args:
            query (str): Search query
            num_results (int, optional): Number of results to return
            site (str, optional): Site to search within (e.g. 'site:example.com')
            
        Returns:
            list: List of search results with extracted content
        """
        logger.info(f"Searching web for: {query}")
        
        # This is a placeholder - in a real implementation this would use a search API
        logger.warning("WebScraper.search_with_extraction() is a placeholder method")
        logger.warning("To use real search functionality, implement a search API integration")
        
        # For demo purposes, we'll just return some dummy results
        # In a real implementation, this would use a search API and then fetch and extract content from the results
        return [
            {
                'title': 'Placeholder Search Result 1',
                'url': 'https://example.com/result1',
                'snippet': f'This is a placeholder result for the query: {query}',
                'content': 'In a real implementation, this would contain extracted content from the page.'
            },
            {
                'title': 'Placeholder Search Result 2',
                'url': 'https://example.com/result2',
                'snippet': f'Another placeholder result for: {query}',
                'content': 'In a real implementation, this would contain extracted content from the page.'
            }
        ]
    
    def close(self):
        """Close the session."""
        if self.session:
            self.session.close()

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
    
    scraper = WebScraper()
    
    try:
        # Test fetching a URL
        result = scraper.fetch_url("https://example.com")
        
        print("\nFetched URL:")
        print(f"Title: {result.get('title', 'No title')}")
        print(f"Content type: {result.get('content_type', 'Unknown')}")
        print(f"Text length: {len(result.get('text', ''))}")
        print(f"Links: {len(result.get('links', []))}")
        
        print("\nText preview:")
        print(result.get('text', 'No text')[:500] + "...")
    finally:
        scraper.close() 