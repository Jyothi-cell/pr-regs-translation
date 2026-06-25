import os
import requests
import json
from anthropic import Anthropic
from typing import Dict, Optional


def get_anthropic_credentials(workspace_id: str = None, model_name: str = None) -> Dict:
    """
    Get Anthropic credentials from the AI Platform.
    
    Args:
        workspace_id: Workspace ID from the workspace console
        model_name: Name of the model to use
        
    Returns:
        dict: Credentials and headers for Anthropic API
    """
    workspace_id = workspace_id or os.getenv("WORKSPACE_ID", "PdfcomparezUzL")
    model_name = model_name or os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
    
    payload = {
        "workspace_id": workspace_id
    }

    url = "https://aiplatform.gcs.int.thomsonreuters.com/v1/anthropic/token"
    
    try:
        # Make the request to get the credentials
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        credentials = resp.json()
        
        print(f"API Response: {credentials}")  # Debug logging

        # Try multiple possible field names for the API key
        api_key = None
        if "anthropic_api_key" in credentials:
            api_key = credentials["anthropic_api_key"]
        elif "anthropic_key" in credentials:
            api_key = credentials["anthropic_key"]
        elif "api_key" in credentials:
            api_key = credentials["api_key"]
        elif "token" in credentials:
            api_key = credentials["token"]
        
        if not api_key:
            raise Exception(f"No valid API key found in response. Available fields: {list(credentials.keys())}")
            
        return {
            "api_key": api_key,
            "model_name": model_name,
            "workspace_id": workspace_id,
            "credentials": credentials
        }
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to AI Platform: {e}")
    except Exception as e:
        raise Exception(f"Failed to retrieve Anthropic credentials: {e}")


def create_anthropic_client(credentials: Dict) -> Anthropic:
    """
    Initialize the Anthropic client with the provided credentials.
    
    Args:
        credentials: Credentials from get_anthropic_credentials
        
    Returns:
        Anthropic: Initialized Anthropic client
    """
    return Anthropic(api_key=credentials["api_key"])


def query_anthropic(client: Anthropic, model_name: str, prompt: str, max_tokens: int = 64000) -> str:
    """
    Send a query to the Anthropic model.
    
    Args:
        client: Initialized Anthropic client
        model_name: Name of the model to use
        prompt: Prompt to send to the model
        max_tokens: Maximum tokens in response
        
    Returns:
        str: Response from the model
    """
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text


def main():
    """Test function for Anthropic client"""
    try:
        # Get credentials
        credentials = get_anthropic_credentials()
        
        # Initialize client
        client = create_anthropic_client(credentials)
        
        # Example query
        prompt = "What is the capital of France?"
        response = query_anthropic(client, credentials["model_name"], prompt)
        
        print(response)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()