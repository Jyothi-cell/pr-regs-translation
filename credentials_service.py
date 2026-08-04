# credentials_service.py
# Activity-scoped helper for retrieving encrypted credentials from the
# GCS Credentials microservice under the WORKFLOW_CUSTOM_ACTIVITIES profile.

import os
from dataclasses import dataclass

from cars_http.cars_http_client import CarsHttpClient
from dataclasses_json import LetterCase, Undefined, dataclass_json
from gcs_jsonlogger.ExecutionContextCategory import ExecutionContextCategory
from gcs_jsonlogger.gcs_logger import GcsLogger


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class EncryptedCredential:
    """
    Represents a credential retrieved from the GCS Credentials microservice.
      - user_name : stores the Anthropic workspace_id (safe to log)
      - password  : stores the Anthropic API key — NEVER log this value
    """
    user_name: str
    password: str  # NEVER log this value


class CredentialsService:
    """
    Retrieves encrypted credentials from the GCS Credentials microservice.
    Uses the WORKFLOW_CUSTOM_ACTIVITIES profile.

    The base URL defaults to the QA environment endpoint and can be overridden
    via the CREDENTIALS_SERVICE_BASE_URL environment variable if injected by
    the platform in future environments.
    """

    # Profile key required by the GCS Credentials microservice
    PROFILE_KEY = "WORKFLOW_CUSTOM_ACTIVITIES"

    def __init__(self, cars_http_client: CarsHttpClient, logger: GcsLogger):
        self._cars_http_client = cars_http_client
        self._logger = logger

        # Use os.getenv with a default fallback to the known QA base URL.
        # This matches the documented platform pattern exactly.
        self._credentials_base_url = os.getenv(
            "CREDENTIALS_SERVICE_BASE_URL",
            "https://credentials-qa.gcs.int.thomsonreuters.com"
        )
        self._logger.info(
            f"CredentialsService initialized. Base URL: {self._credentials_base_url}"
        )

    def get_encrypted_credential(self, credentials_uuid: str) -> EncryptedCredential:
        """
        Retrieve an encrypted credential by its UUID.

        Args:
            credentials_uuid: UUID of the stored credential.

        Returns:
            EncryptedCredential containing user_name (workspace_id) and
            password (api_key — NEVER log).

        Raises:
            ValueError: If credentials_uuid is empty/None.
            Exception:  On unexpected response from the credentials microservice.
        """
        if not credentials_uuid:
            raise ValueError("credentials_uuid must be provided and non-empty")

        # Correct URL pattern as per platform documentation
        url = (
            f"{self._credentials_base_url}"
            f"/v1/credential-profiles/{self.PROFILE_KEY}"
            f"/encrypted-credentials/{credentials_uuid}"
        )

        self._logger.info(
            f"Fetching credential UUID: {credentials_uuid} "
            f"from profile: {self.PROFILE_KEY}"
        )

        # CarsHttpClient.get() returns the parsed dict directly — not a response object.
        # Must pass execution_context_category and includePassword param.
        resp: dict = self._cars_http_client.get(
            url,
            execution_context_category=ExecutionContextCategory.OPTIONAL_BASE_ONLY.value,
            params={"includePassword": "true"}
        )

        credential = EncryptedCredential.from_dict(resp)

        # Log user_name only — NEVER log the password
        self._logger.info(
            f"Credential retrieved successfully. user_name: {credential.user_name}"
        )
        return credential
