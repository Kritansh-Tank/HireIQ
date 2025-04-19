"""
API Tool

This module provides API integration functionality for the agent system.
"""

import logging
import requests
import json
import time
import os
import hashlib
from urllib.parse import urljoin
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config

logger = logging.getLogger(__name__)

class APITool:
    """API tool for interacting with external APIs."""
    
    def __init__(self, cache_dir=None, timeout=None):
        """Initialize the API tool.
        
        Args:
            cache_dir (str, optional): Directory to cache API responses
            timeout (int, optional): Timeout in seconds for API calls
        """
        self.cache_dir = cache_dir or os.path.join(config.TOOLS_DIR, "api_cache")
        self.timeout = timeout or 30
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create session for requests
        self.session = requests.Session()
        
        # API configurations - each key is an API name, value is a config dict
        self.api_configs = {
            # Example: 'github': {'base_url': 'https://api.github.com', 'auth_type': 'bearer', 'token': '...'}
        }
        
        logger.info("Initialized API tool")
    
    def register_api(self, name, base_url, auth_type=None, auth_params=None, headers=None):
        """Register an API configuration.
        
        Args:
            name (str): API name
            base_url (str): Base URL for the API
            auth_type (str, optional): Authentication type (e.g. 'bearer', 'basic', 'api_key')
            auth_params (dict, optional): Authentication parameters
            headers (dict, optional): Default headers for requests
            
        Returns:
            bool: True if registration was successful
        """
        try:
            if not base_url:
                logger.error(f"Invalid base URL for API: {name}")
                return False
            
            config = {
                'base_url': base_url,
                'auth_type': auth_type,
                'auth_params': auth_params or {},
                'headers': headers or {}
            }
            
            self.api_configs[name] = config
            logger.info(f"Registered API: {name} ({base_url})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering API {name}: {str(e)}")
            return False
    
    def _get_cache_path(self, api_name, endpoint, method, data_hash):
        """Get the cache path for an API request.
        
        Args:
            api_name (str): API name
            endpoint (str): API endpoint
            method (str): HTTP method
            data_hash (str): Hash of request data
            
        Returns:
            str: Cache path
        """
        # Create a hash of the request details
        request_key = f"{api_name}_{endpoint}_{method}_{data_hash}"
        request_hash = hashlib.md5(request_key.encode()).hexdigest()
        
        return os.path.join(self.cache_dir, f"{request_hash}.json")
    
    def _prepare_request(self, api_name, endpoint, method, params=None, data=None, json_data=None, headers=None):
        """Prepare a request to an API.
        
        Args:
            api_name (str): API name
            endpoint (str): API endpoint
            method (str): HTTP method
            params (dict, optional): Query parameters
            data (dict, optional): Form data
            json_data (dict, optional): JSON data
            headers (dict, optional): Request headers
            
        Returns:
            tuple: (URL, headers, request_data_hash)
        """
        # Check if API is registered
        if api_name not in self.api_configs:
            raise ValueError(f"API not registered: {api_name}")
        
        api_config = self.api_configs[api_name]
        
        # Build URL
        base_url = api_config['base_url']
        url = urljoin(base_url, endpoint)
        
        # Build headers
        request_headers = {}
        request_headers.update(api_config.get('headers', {}))
        if headers:
            request_headers.update(headers)
        
        # Add authentication
        auth_type = api_config.get('auth_type')
        auth_params = api_config.get('auth_params', {})
        
        if auth_type == 'bearer':
            token = auth_params.get('token')
            if token:
                request_headers['Authorization'] = f"Bearer {token}"
        elif auth_type == 'basic':
            username = auth_params.get('username')
            password = auth_params.get('password')
            if username and password:
                auth = requests.auth.HTTPBasicAuth(username, password)
                self.session.auth = auth
        elif auth_type == 'api_key':
            key_name = auth_params.get('key_name', 'api_key')
            key_value = auth_params.get('key_value')
            key_in = auth_params.get('key_in', 'query')
            
            if key_value:
                if key_in == 'query':
                    if not params:
                        params = {}
                    params[key_name] = key_value
                elif key_in == 'header':
                    request_headers[key_name] = key_value
        
        # Create a hash of the request data for caching
        data_dict = {
            'params': params or {},
            'data': data or {},
            'json': json_data or {}
        }
        data_hash = hashlib.md5(json.dumps(data_dict, sort_keys=True).encode()).hexdigest()
        
        return url, request_headers, data_hash
    
    def call_api(self, api_name, endpoint, method='GET', params=None, data=None, json_data=None, 
                headers=None, use_cache=True, force_refresh=False):
        """Call an API.
        
        Args:
            api_name (str): API name
            endpoint (str): API endpoint
            method (str, optional): HTTP method
            params (dict, optional): Query parameters
            data (dict, optional): Form data
            json_data (dict, optional): JSON data
            headers (dict, optional): Request headers
            use_cache (bool, optional): Whether to use cached responses
            force_refresh (bool, optional): Whether to force refresh the cache
            
        Returns:
            dict: API response
        """
        try:
            # Prepare request
            url, request_headers, data_hash = self._prepare_request(
                api_name, endpoint, method, params, data, json_data, headers
            )
            
            # Check cache if enabled and not forcing refresh
            if use_cache and not force_refresh and method.upper() == 'GET':
                cache_path = self._get_cache_path(api_name, endpoint, method, data_hash)
                
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            cached_data = json.load(f)
                            logger.debug(f"Loaded API response from cache: {api_name} {endpoint}")
                            return cached_data
                    except Exception as e:
                        logger.warning(f"Error loading cached API response: {str(e)}")
            
            # Make the request
            logger.info(f"Calling API: {method} {url}")
            start_time = time.time()
            
            response = self.session.request(
                method=method,
                url=url,
                headers=request_headers,
                params=params,
                data=data,
                json=json_data,
                timeout=self.timeout
            )
            
            # Raise for status
            response.raise_for_status()
            
            # Try to parse JSON response
            try:
                result = response.json()
            except ValueError:
                # Not JSON, return text
                result = {'text': response.text}
            
            end_time = time.time()
            logger.debug(f"API call completed in {end_time - start_time:.2f} seconds: {method} {url}")
            
            # Cache the result if enabled
            if use_cache and method.upper() == 'GET':
                cache_path = self._get_cache_path(api_name, endpoint, method, data_hash)
                
                try:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                        logger.debug(f"Cached API response: {api_name} {endpoint}")
                except Exception as e:
                    logger.warning(f"Error caching API response: {str(e)}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling API {api_name} {endpoint}: {str(e)}")
            
            # Create error result
            error_result = {
                'error': str(e),
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
                'timestamp': time.time()
            }
            
            return error_result
        except Exception as e:
            logger.error(f"Unexpected error calling API {api_name} {endpoint}: {str(e)}")
            return {'error': str(e)}
    
    def get_job_listings(self, query, location=None, job_type=None, limit=10):
        """Get job listings from a job search API.
        
        This is a placeholder method. In a real implementation, this would
        use a job search API like Indeed, LinkedIn, or a custom API.
        
        Args:
            query (str): Job search query
            location (str, optional): Job location
            job_type (str, optional): Job type (e.g. 'full-time', 'part-time')
            limit (int, optional): Maximum number of results to return
            
        Returns:
            list: List of job listings
        """
        logger.info(f"Searching for jobs: {query} in {location}")
        
        # This is a placeholder - in a real implementation this would use a job search API
        logger.warning("APITool.get_job_listings() is a placeholder method")
        logger.warning("To use real job search functionality, implement a job search API integration")
        
        # For demo purposes, we'll just return some dummy results
        return [
            {
                'title': f'Senior {query} Developer',
                'company': 'Example Corp',
                'location': location or 'Remote',
                'type': job_type or 'Full-time',
                'description': f'We are looking for a Senior {query} Developer to join our team.',
                'url': 'https://example.com/jobs/1'
            },
            {
                'title': f'{query} Engineer',
                'company': 'Sample Inc',
                'location': location or 'New York, NY',
                'type': job_type or 'Full-time',
                'description': f'Join our team as a {query} Engineer and work on exciting projects.',
                'url': 'https://example.com/jobs/2'
            }
        ][:limit]
    
    def get_company_info(self, company_name):
        """Get information about a company.
        
        This is a placeholder method. In a real implementation, this would
        use a company information API like Clearbit, LinkedIn, or a custom API.
        
        Args:
            company_name (str): Company name
            
        Returns:
            dict: Company information
        """
        logger.info(f"Getting company info: {company_name}")
        
        # This is a placeholder - in a real implementation this would use a company info API
        logger.warning("APITool.get_company_info() is a placeholder method")
        logger.warning("To use real company info functionality, implement a company info API integration")
        
        # For demo purposes, we'll just return some dummy results
        return {
            'name': company_name,
            'domain': f"{company_name.lower().replace(' ', '')}.com",
            'industry': 'Technology',
            'founded': 2010,
            'employees': 500,
            'description': f"{company_name} is a leading technology company."
        }
    
    def close(self):
        """Close the API tool and free resources."""
        logger.info("Closing API tool")
        
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
    
    api_tool = APITool()
    
    # Register GitHub API
    api_tool.register_api(
        name='github',
        base_url='https://api.github.com',
        auth_type='bearer',
        auth_params={'token': 'YOUR_GITHUB_TOKEN_HERE'},
        headers={'Accept': 'application/vnd.github.v3+json'}
    )
    
    try:
        # Search for jobs (placeholder)
        jobs = api_tool.get_job_listings('Python', location='Remote')
        print("\nJob Listings:")
        for job in jobs:
            print(f"- {job['title']} at {job['company']} ({job['location']})")
        
        # Get company info (placeholder)
        company = api_tool.get_company_info('Example Corp')
        print("\nCompany Info:")
        print(f"- {company['name']} ({company['domain']})")
        print(f"- {company['industry']} | Founded {company['founded']} | {company['employees']} employees")
        print(f"- {company['description']}")
        
        # Comment out the GitHub API call unless you have a valid token
        # Uncomment to try with a valid token
        # github_user = api_tool.call_api('github', 'user')
        # print("\nGitHub User:")
        # print(f"- {github_user.get('login')} ({github_user.get('name')})")
        # print(f"- {github_user.get('bio')}")
    
    finally:
        api_tool.close() 