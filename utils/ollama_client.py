"""
Ollama Client Utility

This module provides a client for interacting with Ollama's API for on-premise LLM access.
"""

import json
import logging
import time
import httpx
import requests
import os
import hashlib
import pickle
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API for on-premise LLM access."""
    
    def __init__(self, model=None, base_url=None, timeout=None, cache_dir=None, use_cache=None):
        """Initialize the Ollama client.
        
        Args:
            model (str, optional): The Ollama model to use.
            base_url (str, optional): Base URL for the Ollama API.
            timeout (int, optional): Timeout in seconds for API calls.
            cache_dir (str, optional): Directory to cache LLM responses.
            use_cache (bool, optional): Whether to use caching.
        """
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.model = model or config.OLLAMA_LLM_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT
        self.http_client = httpx.Client(timeout=self.timeout)
        
        # Caching settings
        self.use_cache = use_cache if use_cache is not None else config.ENABLE_CACHING
        self.cache_dir = cache_dir or config.LLM_CACHE_DIR
        
        # Create cache directory if using cache
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.debug(f"Using LLM response cache at {self.cache_dir}")
        
        # Check if Ollama is available
        self._check_availability()
    
    def _check_availability(self):
        """Check if Ollama API is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Ollama API is available at {self.base_url}")
            
            # Check if the model is available
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            
            if not model_names:
                logger.warning("No models found in Ollama")
            elif self.model not in model_names:
                logger.warning(f"Model {self.model} not found in available models: {model_names}")
                logger.warning(f"Attempting to pull the model {self.model}...")
                self._pull_model()
            else:
                logger.info(f"Using Ollama model: {self.model}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API is not available: {str(e)}")
            logger.warning("Make sure Ollama is running on your system.")
            logger.warning("You can download Ollama from: https://ollama.ai/")
            logger.warning("Start Ollama before continuing.")
    
    def _pull_model(self):
        """Pull the model from Ollama if it doesn't exist."""
        try:
            logger.info(f"Pulling model {self.model} from Ollama...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=300  # Longer timeout for model pulling
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model {self.model}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to pull model {self.model}: {str(e)}")
    
    def _compute_cache_key(self, data_dict):
        """Compute a cache key for the given data.
        
        Args:
            data_dict (dict): Dictionary containing request data
            
        Returns:
            str: Cache key
        """
        # Create a deterministic string representation of the dictionary
        data_str = json.dumps(data_dict, sort_keys=True)
        
        # Create a hash of the string
        hash_obj = hashlib.md5(data_str.encode())
        return hash_obj.hexdigest()
    
    def _get_from_cache(self, cache_key):
        """Get data from cache.
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            Any: Cached data or None if not found
        """
        if not self.use_cache:
            return None
            
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Error loading cache: {str(e)}")
        
        logger.debug(f"Cache miss: {cache_key}")
        return None
    
    def _save_to_cache(self, cache_key, data):
        """Save data to cache.
        
        Args:
            cache_key (str): Cache key
            data (Any): Data to cache
        """
        if not self.use_cache:
            return
            
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
    
    def generate(self, prompt, system=None, max_tokens=None, temperature=0.7, top_p=0.9):
        """Generate text using the Ollama model.
        
        Args:
            prompt (str): The prompt to generate text from
            system (str, optional): System message to guide the model's behavior
            max_tokens (int, optional): Maximum number of tokens to generate
            temperature (float, optional): Sampling temperature (0.0 to 1.0)
            top_p (float, optional): Top-p sampling parameter
            
        Returns:
            str: Generated text
        """
        # Check cache first if caching is enabled and temperature is low
        # We only cache deterministic generations (low temperature)
        should_cache = self.use_cache and temperature < 0.3
        
        if should_cache:
            cache_data = {
                "type": "generate",
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            cache_key = self._compute_cache_key(cache_data)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result is not None:
                logger.debug("Using cached LLM generation result")
                return cached_result
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p
            }
        }
        
        if system:
            payload["system"] = system
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        # Initialize retry counter
        retries = 0
        max_retries = config.OLLAMA_MAX_RETRIES
        retry_delay = config.OLLAMA_RETRY_DELAY
        
        while retries <= max_retries:
            try:
                start_time = time.time()
                logger.debug(f"Generating with Ollama model {self.model}: {prompt[:100]}...")
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                generated_text = result.get("response", "")
                
                end_time = time.time()
                logger.debug(f"Generation completed in {end_time - start_time:.2f} seconds")
                
                # Cache the result if appropriate
                if should_cache:
                    self._save_to_cache(cache_key, generated_text)
                
                # Return successful result
                return generated_text
                    
            except requests.exceptions.Timeout:
                retries += 1
                if retries <= max_retries:
                    # Log timeout and retry
                    logger.warning(f"Timeout occurred while generating text. Retrying ({retries}/{max_retries})...")
                    # Wait before retrying, increasing delay with each retry
                    time.sleep(retry_delay * retries)
                else:
                    # Log failure after all retries
                    logger.error(f"Failed to generate text after {max_retries} retries due to timeout.")
                    return f"Error: Failed to generate text with Ollama (timeout after {max_retries} retries)"
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error generating text with Ollama: {str(e)}")
                return f"Error: Failed to generate text with Ollama ({str(e)})"
    
    def chat(self, messages, max_tokens=None, temperature=0.7, top_p=0.9):
        """Have a chat conversation with the Ollama model.
        
        Args:
            messages (list): List of message dicts with 'role' and 'content' keys
                             Roles can be 'system', 'user', or 'assistant'
            max_tokens (int, optional): Maximum number of tokens to generate
            temperature (float, optional): Sampling temperature (0.0 to 1.0)
            top_p (float, optional): Top-p sampling parameter
            
        Returns:
            str: Generated response
        """
        # Check cache first if caching is enabled and temperature is low
        # We only cache deterministic generations (low temperature)
        should_cache = self.use_cache and temperature < 0.3
        
        if should_cache:
            cache_data = {
                "type": "chat",
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            cache_key = self._compute_cache_key(cache_data)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result is not None:
                logger.debug("Using cached LLM chat result")
                return cached_result
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        # Initialize retry counter
        retries = 0
        max_retries = config.OLLAMA_MAX_RETRIES
        retry_delay = config.OLLAMA_RETRY_DELAY
        
        while retries <= max_retries:
            try:
                start_time = time.time()
                logger.debug(f"Chatting with Ollama model {self.model}")
                
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                generated_text = result.get("message", {}).get("content", "")
                
                end_time = time.time()
                logger.debug(f"Chat completed in {end_time - start_time:.2f} seconds")
                
                # Cache the result if appropriate
                if should_cache:
                    self._save_to_cache(cache_key, generated_text)
                
                # Return successful result
                return generated_text
                    
            except requests.exceptions.Timeout:
                retries += 1
                if retries <= max_retries:
                    # Log timeout and retry
                    logger.warning(f"Timeout occurred while chatting. Retrying ({retries}/{max_retries})...")
                    # Wait before retrying, increasing delay with each retry
                    time.sleep(retry_delay * retries)
                else:
                    # Log failure after all retries
                    logger.error(f"Failed to generate chat response after {max_retries} retries due to timeout.")
                    return f"Error: Failed to chat with Ollama (timeout after {max_retries} retries)"
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error chatting with Ollama: {str(e)}")
                return f"Error: Failed to chat with Ollama ({str(e)})"
    
    def clear_cache(self):
        """Clear the LLM response cache."""
        if not self.use_cache:
            logger.debug("Caching is disabled, no cache to clear")
            return
            
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            for cache_file in cache_files:
                os.remove(os.path.join(self.cache_dir, cache_file))
            
            logger.info(f"Cleared {len(cache_files)} cache files from {self.cache_dir}")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    def close(self):
        """Close the HTTP client."""
        if self.http_client:
            self.http_client.close()

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
    print("OLLAMA CLIENT - WITH CACHING DEMO")
    print("=" * 50)
    
    # Create a client with caching enabled
    client = OllamaClient(use_cache=True)
    
    # Test generation
    print("\nFirst generation (should be slower):")
    start_time = time.time()
    response = client.generate(
        "What are the key skills needed for a software engineer in 2023?",
        temperature=0.1  # Low temperature for deterministic results
    )
    print(f"  Time taken: {time.time() - start_time:.2f} seconds")
    print(f"  Response: {response[:150]}...")
    
    # Test cached generation
    print("\nSecond generation (should be faster due to caching):")
    start_time = time.time()
    response2 = client.generate(
        "What are the key skills needed for a software engineer in 2023?",
        temperature=0.1  # Low temperature for deterministic results
    )
    print(f"  Time taken: {time.time() - start_time:.2f} seconds")
    print(f"  Response: {response2[:150]}...")
    
    # Test chat
    print("\nChat test:")
    chat_response = client.chat([
        {"role": "system", "content": "You are a helpful AI assistant for job screening."},
        {"role": "user", "content": "What should I look for in a candidate's resume for a data scientist position?"}
    ], temperature=0.1)
    print(chat_response[:150] + "...")
    
    # Display cache stats
    cache_files = os.listdir(client.cache_dir)
    print(f"\nCache status: {len(cache_files)} files in cache")
    
    client.close() 