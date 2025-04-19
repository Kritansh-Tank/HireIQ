"""
Base Agent

This module provides a base agent class that all agents inherit from.
It includes Ollama LLM integration, message handling, and common agent functionality.
"""

import logging
import time
import traceback
import queue
import threading
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config
from utils.ollama_client import OllamaClient
from utils.embeddings import EmbeddingUtility

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base agent class that provides common functionality for all agents."""
    
    def __init__(self, name, model=None, system_prompt=None):
        """Initialize the base agent.
        
        Args:
            name (str): Agent name
            model (str, optional): The Ollama model to use
            system_prompt (str, optional): System prompt to guide the agent's behavior
        """
        self.name = name
        self.model = model or config.OLLAMA_LLM_MODEL
        self.system_prompt = system_prompt or f"You are an AI assistant helping with job screening tasks. Your name is {name}."
        
        # Initialize Ollama client
        self.ollama_client = OllamaClient(model=self.model)
        
        # Initialize embedding utility
        self.embedding_util = EmbeddingUtility()
        
        # Message queue for agent communication
        self.message_queue = queue.Queue(maxsize=config.AGENT_MESSAGE_QUEUE_SIZE)
        
        # Thread for processing messages
        self.message_thread = None
        self.running = False
        
        # Agent state
        self.state = {}
        
        # Tool registry
        self.tools = {}
        
        # Default tools
        self._register_default_tools()
        
        logger.info(f"Initialized agent: {self.name}")
    
    def _register_default_tools(self):
        """Register default tools available to all agents."""
        # Register tools
        self.register_tool("get_embedding", self.embedding_util.get_embedding, 
                         "Generate an embedding for a text")
        self.register_tool("compute_similarity", self.embedding_util.compute_similarity,
                         "Compute similarity between two texts")
        self.register_tool("semantic_search", self.embedding_util.semantic_search,
                         "Perform semantic search over a set of documents")
    
    def register_tool(self, name, function, description):
        """Register a tool that the agent can use.
        
        Args:
            name (str): Tool name
            function (callable): Tool function
            description (str): Tool description
        """
        self.tools[name] = {
            'function': function,
            'description': description
        }
        logger.debug(f"Agent {self.name} registered tool: {name}")
    
    def use_tool(self, tool_name, **kwargs):
        """Use a tool by name.
        
        Args:
            tool_name (str): Tool name
            **kwargs: Tool arguments
            
        Returns:
            Any: Tool result
        """
        if tool_name not in self.tools:
            error_msg = f"Tool not found: {tool_name}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        tool = self.tools[tool_name]
        
        try:
            logger.debug(f"Agent {self.name} using tool: {tool_name}")
            result = tool['function'](**kwargs)
            return result
        except Exception as e:
            error_msg = f"Error using tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": error_msg}
    
    def get_available_tools(self):
        """Get a list of available tools.
        
        Returns:
            list: List of tool info dictionaries
        """
        return [
            {'name': name, 'description': tool['description']}
            for name, tool in self.tools.items()
        ]
    
    def process_with_llm(self, prompt, system=None, max_tokens=1024, temperature=0.7):
        """Process a prompt with the LLM.
        
        Args:
            prompt (str): The prompt to process
            system (str, optional): System message to guide the model's behavior
            max_tokens (int, optional): Maximum number of tokens to generate
            temperature (float, optional): Sampling temperature (0.0 to 1.0)
            
        Returns:
            str: LLM response
        """
        system_message = system or self.system_prompt
        
        try:
            for attempt in range(config.AGENT_MAX_RETRIES):
                try:
                    response = self.ollama_client.generate(
                        prompt=prompt,
                        system=system_message,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response
                except Exception as e:
                    if attempt < config.AGENT_MAX_RETRIES - 1:
                        logger.warning(f"Error processing with LLM (attempt {attempt+1}): {str(e)}")
                        time.sleep(config.AGENT_RETRY_DELAY)
                    else:
                        raise
        except Exception as e:
            logger.error(f"Error processing with LLM: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error: Failed to process with LLM ({str(e)})"
    
    def process_with_chat(self, messages, max_tokens=1024, temperature=0.7):
        """Process a chat conversation with the LLM.
        
        Args:
            messages (list): List of message dicts with 'role' and 'content' keys
            max_tokens (int, optional): Maximum number of tokens to generate
            temperature (float, optional): Sampling temperature (0.0 to 1.0)
            
        Returns:
            str: LLM response
        """
        # Add system message if not present
        if not messages or messages[0].get('role') != 'system':
            messages.insert(0, {'role': 'system', 'content': self.system_prompt})
        
        try:
            for attempt in range(config.AGENT_MAX_RETRIES):
                try:
                    response = self.ollama_client.chat(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response
                except Exception as e:
                    if attempt < config.AGENT_MAX_RETRIES - 1:
                        logger.warning(f"Error processing chat with LLM (attempt {attempt+1}): {str(e)}")
                        time.sleep(config.AGENT_RETRY_DELAY)
                    else:
                        raise
        except Exception as e:
            logger.error(f"Error processing chat with LLM: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error: Failed to process chat with LLM ({str(e)})"
    
    def start_message_processing(self):
        """Start the message processing thread."""
        if self.message_thread is not None and self.message_thread.is_alive():
            logger.warning(f"Agent {self.name} message thread already running")
            return
        
        self.running = True
        self.message_thread = threading.Thread(target=self._process_messages)
        self.message_thread.daemon = True
        self.message_thread.start()
        logger.info(f"Agent {self.name} message thread started")
    
    def stop_message_processing(self):
        """Stop the message processing thread."""
        self.running = False
        
        if self.message_thread is not None:
            self.message_thread.join(timeout=5.0)
            logger.info(f"Agent {self.name} message thread stopped")
    
    def _process_messages(self):
        """Process messages from the queue."""
        logger.info(f"Agent {self.name} starting message processing")
        
        while self.running:
            try:
                # Get message from queue, block for 1 second
                try:
                    message = self.message_queue.get(block=True, timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process message
                logger.debug(f"Agent {self.name} processing message: {message.get('type', 'unknown')}")
                
                response = self.handle_message(message)
                
                # Send response if needed
                if response and message.get('reply_to'):
                    self.send_message(
                        message_type='response',
                        content=response,
                        recipient=message['sender'],
                        correlation_id=message.get('correlation_id')
                    )
                
                # Mark message as processed
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing message in agent {self.name}: {str(e)}")
                logger.error(traceback.format_exc())
    
    def handle_message(self, message):
        """Handle a message.
        
        This method should be overridden by subclasses.
        
        Args:
            message (dict): Message to handle
            
        Returns:
            Any: Response to the message
        """
        # Default implementation just logs the message
        logger.debug(f"Agent {self.name} received message: {message}")
        
        message_type = message.get('type', 'unknown')
        content = message.get('content')
        
        if message_type == 'query':
            # Process query with LLM
            if isinstance(content, str):
                return self.process_with_llm(content)
            elif isinstance(content, list):
                return self.process_with_chat(content)
        
        if message_type == 'tool_use':
            # Use a tool
            tool_name = content.get('tool_name')
            tool_args = content.get('tool_args', {})
            
            if tool_name:
                return self.use_tool(tool_name, **tool_args)
        
        # For other message types, return None
        return None
    
    def send_message(self, message_type, content, recipient, correlation_id=None):
        """Send a message to another agent.
        
        Args:
            message_type (str): Type of message
            content (Any): Message content
            recipient (str): Recipient agent name
            correlation_id (str, optional): Correlation ID for message tracking
        """
        # Create message
        message = {
            'type': message_type,
            'content': content,
            'sender': self.name,
            'recipient': recipient,
            'timestamp': time.time()
        }
        
        if correlation_id:
            message['correlation_id'] = correlation_id
        
        # Find the recipient agent in the agent registry
        # This is a simple implementation - in a real system, you'd use a message broker
        from agents import agent_registry
        
        recipient_agent = agent_registry.get_agent(recipient)
        
        if recipient_agent:
            recipient_agent.message_queue.put(message)
            logger.debug(f"Agent {self.name} sent message to {recipient}: {message_type}")
        else:
            logger.error(f"Agent {self.name} failed to send message: recipient {recipient} not found")
    
    def query(self, recipient, content, timeout=30.0):
        """Query another agent and wait for a response.
        
        Args:
            recipient (str): Recipient agent name
            content (Any): Query content
            timeout (float, optional): Timeout in seconds
            
        Returns:
            Any: Response from the recipient agent
        """
        correlation_id = str(time.time())
        response_queue = queue.Queue()
        
        # Register temporary handler for the response
        def response_handler(message):
            if message.get('correlation_id') == correlation_id:
                response_queue.put(message.get('content'))
        
        # Send the query
        self.send_message(
            message_type='query',
            content=content,
            recipient=recipient,
            correlation_id=correlation_id
        )
        
        # Wait for response
        try:
            response = response_queue.get(block=True, timeout=timeout)
            return response
        except queue.Empty:
            logger.error(f"Timeout waiting for response from {recipient}")
            return None
    
    def close(self):
        """Close the agent and free resources."""
        logger.info(f"Closing agent: {self.name}")
        
        # Stop message processing
        self.stop_message_processing()
        
        # Close Ollama client
        if self.ollama_client:
            self.ollama_client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Agent registry for storing and retrieving agents
class AgentRegistry:
    """Registry for storing and retrieving agents."""
    
    def __init__(self):
        """Initialize the agent registry."""
        self.agents = {}
    
    def register_agent(self, agent):
        """Register an agent.
        
        Args:
            agent (BaseAgent): Agent to register
        """
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def get_agent(self, name):
        """Get an agent by name.
        
        Args:
            name (str): Agent name
            
        Returns:
            BaseAgent: Agent instance or None if not found
        """
        return self.agents.get(name)
    
    def get_all_agents(self):
        """Get all registered agents.
        
        Returns:
            dict: Dictionary of agent names to agent instances
        """
        return self.agents
    
    def close_all(self):
        """Close all registered agents."""
        for agent in self.agents.values():
            agent.close()
        
        self.agents = {}

# Create global agent registry
agent_registry = AgentRegistry()

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
    
    # Create test agent
    agent = BaseAgent("TestAgent")
    
    # Start message processing
    agent.start_message_processing()
    
    try:
        # Test LLM
        response = agent.process_with_llm("What are the key qualities of a good software engineer?")
        print(f"\nLLM Response:\n{response}")
        
        # Test tools
        similarity = agent.use_tool("compute_similarity", 
                                   text1="Software engineer with Python skills",
                                   text2="Python developer experienced in Django")
        print(f"\nSimilarity: {similarity:.4f}")
    finally:
        # Close agent
        agent.close() 