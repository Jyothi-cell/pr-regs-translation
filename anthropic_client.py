# anthropic_client.py
# Activity-scoped Anthropic API client helper.
# workspace_id and api_key are ALWAYS passed explicitly from main.py
# via CredentialsService — never read from os.environ here.
# This file is scoped to this activity's workspace only.

import logging
import re
import requests
from anthropic import Anthropic
from typing import Dict, Optional

# Module-level logger — uses structured logging via gcs_jsonlogger
logger = logging.getLogger(__name__)

# Default model name — can be overridden by passing model_name explicitly
# Never read model name from os.environ in this helper
DEFAULT_MODEL_NAME = "claude-sonnet-5"
AI_PLATFORM_TOKEN_URL = "https://aiplatform.gcs.int.thomsonreuters.com/v1/anthropic/token"
WORKSPACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,128}$")


def _sanitize_log_text(text: str) -> str:
    """Best-effort redaction for known secret-like key/value patterns in log text."""
    if not text:
        return ""

    sanitized = str(text)
    patterns = [
        r"(?i)(api[_-]?key\s*[:=]\s*)([^\s,;]+)",
        r"(?i)(token\s*[:=]\s*)([^\s,;]+)",
        r"(?i)(password\s*[:=]\s*)([^\s,;]+)"
    ]
    for pattern in patterns:
        sanitized = re.sub(pattern, r"\1[REDACTED]", sanitized)

    return sanitized


def _validate_workspace_id(workspace_id: str) -> str:
    """Validate workspace ID format and return normalized value."""
    normalized = (workspace_id or "").strip()
    if not normalized:
        raise ValueError(
            "workspace_id must be explicitly provided — never read from environment"
        )

    if not WORKSPACE_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Invalid workspace_id format. "
            "Expected 6-128 chars: letters, numbers, underscore, or hyphen."
        )

    return normalized


def create_anthropic_client(api_key: str) -> Anthropic:
    """
    Initialize the Anthropic client with the provided API key.

    api_key is sourced from CredentialsService in main.py:
      - credential.password -> api_key  (NEVER log this value)

    Args:
        api_key: Anthropic API key retrieved from CredentialsService.
                 NEVER log this value.

    Returns:
        Anthropic: Initialized Anthropic client.

    Raises:
        ValueError: If api_key is missing or empty.
    """
    if not api_key:
        raise ValueError(
            "api_key must be explicitly provided — never read from environment"
        )

    # api_key is intentionally NOT logged — it is a secret
    logger.info("Initializing Anthropic client.")
    return Anthropic(api_key=api_key)


def get_anthropic_credentials(
    workspace_id: str,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME
) -> Dict:
    """
    Build a credentials dict.

        workspace_id is sourced from CredentialsService in main.py.
        api_key may come from either:
            1) GCS AI Platform token endpoint (preferred, fetched at runtime), or
            2) credential.password from CredentialsService as fallback.

        Credential mapping from main.py:
      - credential.user_name -> workspace_id  (safe to log)
            - credential.password  -> api_key fallback (NEVER log this value)

    No environment variable fallbacks are used here.

    Args:
        workspace_id: Anthropic workspace ID — safe to log.
        api_key:      Optional fallback Anthropic API key — NEVER log this value.
        model_name:   Model name to use, defaults to DEFAULT_MODEL_NAME.

    Returns:
        Dict containing workspace_id, api_key, and model_name.

    Raises:
        ValueError: If workspace_id is missing.
        Exception:  If runtime token fetch fails and no fallback api_key exists.
    """
    workspace_id = _validate_workspace_id(workspace_id)
    runtime_api_key = None

    # Prefer runtime fetch from GCS AI Platform so temporary keys are refreshed
    # on every activity execution.
    try:
        logger.info(
            f"Fetching Anthropic runtime token from AI Platform for workspace_id: {workspace_id}"
        )
        resp = requests.post(
            AI_PLATFORM_TOKEN_URL,
            json={"workspace_id": workspace_id},
            timeout=(10, 30)
        )
        resp.raise_for_status()
        payload = resp.json() if resp.content else {}

        # Handle potential response field variants defensively.
        runtime_api_key = (
            payload.get("anthropic_api_key")
            or payload.get("anthropic_key")
            or payload.get("api_key")
            or payload.get("token")
        )

        if runtime_api_key:
            logger.info("Retrieved Anthropic runtime token from AI Platform.")
        else:
            logger.warning(
                "AI Platform response did not contain an Anthropic API key field; "
                "falling back to credential password if present."
            )
    except Exception as e:
        logger.warning(
            "AI Platform token fetch failed; falling back to credential password. "
            f"Error: {_sanitize_log_text(str(e))}"
        )

    resolved_api_key = runtime_api_key or api_key
    if not resolved_api_key:
        raise ValueError(
            "No Anthropic API key available. Runtime fetch failed and fallback api_key was not provided."
        )

    return {
        "workspace_id": workspace_id,   # safe to log
        "api_key": resolved_api_key,    # NEVER log this value
        "model_name": model_name
    }


def query_anthropic(
    client: Anthropic,
    model_name: str,
    prompt: str,
    max_tokens: int = 64000
) -> str:
    """
    Send a text query to the Anthropic model and return the response text.

    Args:
        client:     Initialized Anthropic client from create_anthropic_client().
        model_name: Name of the model to use (e.g. "claude-sonnet-5").
        prompt:     Prompt string to send to the model.
        max_tokens: Maximum tokens in response, default 64000.

    Returns:
        str: Text response from the model.

    Raises:
        ValueError: If client, model_name, or prompt are missing.
        Exception:  On unexpected Anthropic API errors.
    """
    if not client:
        raise ValueError("A valid Anthropic client must be provided")
    if not model_name:
        raise ValueError("model_name must be provided")
    if not prompt:
        raise ValueError("prompt must be provided")

    logger.info(f"Sending query to Anthropic model: {model_name}")

    # Send the message to the Anthropic API
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    logger.info("Anthropic query completed successfully.")

    # Return the text content from the first response block
    return response.content[0].text


def query_anthropic_with_image(
    client: Anthropic,
    model_name: str,
    prompt: str,
    image_data: str,
    media_type: str = "image/png",
    max_tokens: int = 64000
) -> str:
    """
    Send an image-based query to the Anthropic model (Claude Vision).

    Used by pdf_processor.py to send rendered PDF page images to Claude
    for HTML conversion.

    Args:
        client:     Initialized Anthropic client from create_anthropic_client().
        model_name: Name of the model to use (e.g. "claude-sonnet-5").
        prompt:     Text prompt to accompany the image.
        image_data: Base64-encoded image string.
        media_type: MIME type of the image, default "image/png".
        max_tokens: Maximum tokens in response, default 64000.

    Returns:
        str: Text response from the model.

    Raises:
        ValueError: If any required argument is missing.
        Exception:  On unexpected Anthropic API errors.
    """
    if not client:
        raise ValueError("A valid Anthropic client must be provided")
    if not model_name:
        raise ValueError("model_name must be provided")
    if not prompt:
        raise ValueError("prompt must be provided")
    if not image_data:
        raise ValueError("image_data must be provided")

    logger.info(
        f"Sending image-based query to Anthropic model: {model_name}, "
        f"media_type: {media_type}"
    )

    # Build the multipart message with image and text content blocks
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        # Image block — base64 encoded page image
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        # Text prompt block
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    logger.info("Anthropic image-based query completed successfully.")

    # Return the text content from the first response block
    return response.content[0].text
