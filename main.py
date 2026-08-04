# main.py
# CustomActivityRunner for image-based PDF to HTML conversion using Claude Vision API.
# Credentials (workspace_id, api_key) are retrieved securely from the GCS
# Credentials microservice — never from environment files or hardcoded values.

import os
import re
import uuid
import tempfile
import traceback
import logging
from dataclasses import dataclass
from typing import Optional, List

# Core activity SDK imports
from activities_shared.models.custom_activity_request import CustomActivityRequest
from activities_shared.models.custom_activity_response import (
    CustomActivityResponse, 
    WorkflowResultCode
)
from activities_shared.util.base_custom_activity_runner import BaseCustomActivityRunner

# Shared platform services — use these; never make raw HTTP calls for ledger
from activities_shared.services.execution_journal_service import ExecutionJournalService
from activities_shared.models.rendition_criteria import RenditionCriteria

# Storage service for rendition read/write
from cars_storage.random_access.rendition import RenditionStorageService
from cars_storage.execution_journal.data_models import (
    ExecutionLedgersMetadata,
    LedgerItemsRenditions
)
from cars_storage.random_access.rendition.data_models import PostRenditionParameters

from dataclasses_json import dataclass_json, LetterCase
from gcs_common.exceptions import ContextualException
from gcs_common.reason_codes import ReasonCode

# Activity-scoped credentials helper (credentials_service.py in this workspace)
from credentials_service import CredentialsService, EncryptedCredential

# Activity-scoped image-based PDF converter (pdf_processor.py in this workspace)
from pdf_processor import ImageBasedPDFConverter


# Credential UUID configured for this activity implementation.
# Populate this with the UUID created in Credentials Service under
# WORKFLOW_CUSTOM_ACTIVITIES profile.
WORKSPACE_CREDENTIAL_UUID = "9bf96c65-4a22-4d79-b145-fe472b7cfc06"


# ---------------------------------------------------------------------------
# Filename safety helpers
# ---------------------------------------------------------------------------

def _is_unsafe_filename(file_name: str) -> bool:
    """Return True if the filename contains unsafe path characters."""
    if not file_name:
        return True
    normalized = file_name.strip()
    if not normalized:
        return True
    if "/" in normalized or "\\" in normalized or ".." in normalized:
        return True
    return False


def _sanitize_filename(file_name: str) -> str:
    """Strip unsafe characters from a filename, returning a safe fallback."""
    base_name = os.path.basename((file_name or "").strip())
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base_name)
    safe = safe.strip("._-")
    return safe or "document"


def _create_secure_temp_path(*, prefix: str, suffix: str, dir_path: str = "/tmp") -> str:
    """Create a secure temporary file with restricted permissions (0o600)."""
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=dir_path)
    try:
        try:
            os.fchmod(fd, 0o600)
        except AttributeError:
            os.chmod(temp_path, 0o600)
    finally:
        os.close(fd)
    return temp_path


def _secure_delete_file(file_path: str) -> None:
    """Overwrite file contents with zeros before deletion (best-effort)."""
    if not file_path or not os.path.isfile(file_path):
        return
    try:
        file_size = os.path.getsize(file_path)
        if file_size > 0:
            with open(file_path, "r+b") as f:
                chunk = b"\x00" * 1024 * 1024
                remaining = file_size
                while remaining > 0:
                    write_size = min(len(chunk), remaining)
                    f.write(chunk[:write_size])
                    remaining -= write_size
                f.flush()
                os.fsync(f.fileno())
    except Exception:
        pass
    os.remove(file_path)


# ---------------------------------------------------------------------------
# Optional dependency: PyMuPDF (HTML to PDF)
# ---------------------------------------------------------------------------

try:
    import fitz  # PyMuPDF
    HTML_TO_PDF_AVAILABLE = True
except ImportError:
    HTML_TO_PDF_AVAILABLE = False
    fitz = None

# Optional dependency: BeautifulSoup (HTML cleanup)
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ProcessingConfig:
    """
    Configuration for document processing.

    workspace_id and api_key are sourced securely from the GCS Credentials
    microservice at runtime — never from environment files or hardcoded values.

    Rendition type/subtype codes are driven from execution attributes to allow
    reuse across different rendition configurations without code changes.
    """
    image_dpi: int = 450
    generate_pdf_output: bool = False
    workspace_id: str = ""
    api_key: str = ""                        # NEVER log this value
    input_rendition_type_code: str = "SOURCE"
    input_rendition_sub_type_code: str = "ACQUIRED"
    output_rendition_type_code: str = "SOURCE"
    output_rendition_sub_type_code: str = "CONVERTED"


# ---------------------------------------------------------------------------
# HTML to PDF converter (PyMuPDF Story API)
# ---------------------------------------------------------------------------

class HTMLToPDFConverter:
    """
    Converts textual HTML (produced by Claude Vision) to a PDF document
    using PyMuPDF's Story API.
    """

    def __init__(self, logger=None):
        # Use injected platform logger or fall back to standard logger
        self._logger = logger or logging.getLogger(__name__)
        if not HTML_TO_PDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF (fitz) is not available. Cannot create HTMLToPDFConverter."
            )

    def convert_html_to_pdf(self, html_path: str) -> bytes:
        """
        Convert an HTML file to PDF bytes using PyMuPDF.

        Args:
            html_path: Path to the HTML file on disk.

        Returns:
            Generated PDF content as bytes.
        """
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            self._logger.info(
                "Converting text-based HTML to PDF using PyMuPDF Story API."
            )
            pdf_bytes = self._render_html_with_story(html_content)

            if not pdf_bytes:
                raise Exception("PDF generation with Story API produced no output.")

            self._logger.info(
                f"PDF successfully generated from HTML: {len(pdf_bytes)} bytes"
            )
            return pdf_bytes

        except Exception as e:
            error_msg = f"Error in HTML-to-PDF conversion: {str(e)}"
            self._logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return self._create_error_pdf(error_msg)

    def _render_html_with_story(self, html_content: str) -> Optional[bytes]:
        """
        Render HTML content to PDF using the PyMuPDF Story and DocumentWriter API.

        Args:
            html_content: HTML string to render.

        Returns:
            PDF bytes, or None on failure.
        """
        temp_output_path = _create_secure_temp_path(
            prefix="prregs_story_", suffix=".pdf"
        )
        try:
            # Story API works best with content inside the <body> tag
            if BS4_AVAILABLE:
                soup = BeautifulSoup(html_content, 'html.parser')
                body_content = (
                    soup.body.decode_contents() if soup.body else html_content
                )
            else:
                body_content = html_content

            story = fitz.Story(html=body_content)
            writer = fitz.DocumentWriter(temp_output_path)

            # Use standard letter page size
            page_rect = fitz.paper_rect("letter")
            more = 1
            while more:
                device = writer.begin_page(page_rect)
                more, _ = story.place(page_rect)
                story.draw(device)
                writer.end_page()

            writer.close()

            with open(temp_output_path, 'rb') as f:
                pdf_bytes = f.read()

            self._logger.info(
                f"PyMuPDF Story API generated a PDF of {len(pdf_bytes)} bytes."
            )
            return pdf_bytes

        except Exception as e:
            self._logger.error(
                f"PyMuPDF Story API failed: {e}\n{traceback.format_exc()}"
            )
            return None
        finally:
            # Always clean up the temp story output file
            if os.path.exists(temp_output_path):
                try:
                    _secure_delete_file(temp_output_path)
                except Exception:
                    pass

    def _create_error_pdf(self, error_message: str) -> bytes:
        """
        Create a minimal PDF containing an error message as a last-resort fallback.

        Args:
            error_message: Error text to embed in the PDF.

        Returns:
            PDF bytes.
        """
        try:
            doc = fitz.open()
            page = doc.new_page()
            rect = page.rect + (50, 50, -50, -50)
            page.insert_textbox(
                rect,
                f"PDF Generation Failed:\n\n{error_message}",
                fontsize=12,
                fontname="helv"
            )
            pdf_bytes = doc.tobytes()
            doc.close()
            return pdf_bytes
        except Exception as e:
            raise Exception(f"All PDF creation methods failed. Final error: {e}")


# ---------------------------------------------------------------------------
# Main activity runner
# ---------------------------------------------------------------------------

class CustomActivityRunner(BaseCustomActivityRunner):
    """
    Document processing workflow:
      1. Retrieve credentials securely from GCS Credentials microservice.
      2. pdf_processor.py  -> PDF to HTML using Claude Vision API.
      3. HTMLToPDFConverter -> HTML to PDF using PyMuPDF Story API (optional).
    """

    def run(self, activity_request: CustomActivityRequest) -> CustomActivityResponse:
        """
        Entry point for the custom activity.

        Steps:
          0. Log context and parse config with secure credential retrieval.
          1. Locate and download the input PDF rendition.
          2. Convert PDF to HTML via Claude Vision (image-based).
          3. Post the HTML rendition to storage.
          4. Optionally convert HTML to PDF via PyMuPDF and post to storage.
          5. Clean up temporary files.
        """
        # Log full context at the top for DataDog traceability
        self._logger.info(
            f"Starting script execution for activityExecutionGuid: "
            f"{activity_request.activity_execution_guid}"
        )
        self._logger.info(
            f"Workflow execution ID: {activity_request.workflow_execution_id}, "
            f"execution_attributes: {activity_request.execution_attributes}"
        )
        self._logger.info(f"Received inputs: {activity_request.inputs}")

        temp_files_to_cleanup = []

        try:
            # STEP 0: Parse config and securely retrieve credentials
            config = self._parse_processing_config(activity_request)
            self._logger.info(
                f"Processing configuration: DPI={config.image_dpi}, "
                f"generate_pdf={config.generate_pdf_output}, "
                f"workspace_id={config.workspace_id}, "
                f"input_rendition={config.input_rendition_type_code}/"
                f"{config.input_rendition_sub_type_code}, "
                f"output_rendition={config.output_rendition_type_code}/"
                f"{config.output_rendition_sub_type_code}"
                # api_key is intentionally NOT logged
            )

            storage_service = RenditionStorageService(
                cars_http_client=self._cars_http_client
            )

            # STEP 1: Locate and download the input PDF
            input_rend, input_meta, pdf_data = self._get_input_pdf(
                activity_request, storage_service, config
            )

            # Validate and sanitize the input filename before writing to disk
            original_file_name = str(getattr(input_meta, "file_name", "") or "")
            if _is_unsafe_filename(original_file_name):
                raise ContextualException(
                    "Invalid input filename",
                    reason_code=ReasonCode.GCS_WORKFLOW_INTERNAL_ERROR,
                    reason_text="Input filename contains unsafe path characters"
                )

            safe_file_name = _sanitize_filename(original_file_name)
            _, safe_suffix = os.path.splitext(safe_file_name)
            safe_suffix = safe_suffix if safe_suffix else ".pdf"
            safe_stem = os.path.splitext(safe_file_name)[0][:32] or "document"

            # Write PDF bytes to a secure temporary file
            temp_pdf_path = _create_secure_temp_path(
                prefix=f"prregs_{safe_stem}_", suffix=safe_suffix
            )
            with open(temp_pdf_path, 'wb') as f:
                f.write(pdf_data)

            temp_files_to_cleanup.append(temp_pdf_path)
            self._logger.info(f"Saved input PDF to: {temp_pdf_path}")

            # STEP 2: PDF to HTML (Claude Vision, image-based)
            html_bytes, html_file_path, conversion_stats = \
                self._convert_pdf_to_html_image_based(
                    temp_pdf_path, config, temp_files_to_cleanup
                )

            # STEP 3: Post HTML rendition to storage
            self._logger.info("STEP 3: Posting HTML rendition to storage...")
            html_guid = self._post_html_rendition(
                storage_service, input_rend, input_meta,
                html_bytes, html_file_path, config
            )

            # STEP 4: Optionally convert HTML to PDF and post to storage
            pdf_guid = None
            if config.generate_pdf_output:
                self._logger.info(
                    "STEP 4: Converting HTML to PDF using PyMuPDF Story API..."
                )
                if not HTML_TO_PDF_AVAILABLE:
                    raise ContextualException(
                        "PDF generation requested but PyMuPDF not available",
                        reason_code=ReasonCode.GCS_WORKFLOW_INTERNAL_ERROR,
                        reason_text="PyMuPDF should be available from requirements.txt"
                    )
                pdf_guid = self._convert_html_to_pdf(
                    html_file_path, storage_service, input_rend, input_meta,
                    html_guid, temp_files_to_cleanup
                )
                if not pdf_guid:
                    raise ContextualException(
                        "PDF generation failed",
                        reason_code=ReasonCode.GCS_WORKFLOW_INTERNAL_ERROR,
                        reason_text="HTML to PDF conversion did not produce output"
                    )
            else:
                self._logger.info(
                    "STEP 4: Skipping HTML to PDF conversion (disabled in config)"
                )

            # STEP 5: Clean up all temporary files securely
            self._logger.info("STEP 5: Cleaning up temporary files...")
            self._cleanup_temp_files(temp_files_to_cleanup)

            return self._create_success_response(
                html_guid, pdf_guid, conversion_stats, config
            )

        except ContextualException:
            # Re-raise ContextualExceptions directly — they carry user-facing reason codes
            self._cleanup_temp_files(temp_files_to_cleanup)
            raise

        except Exception as e:
            self._logger.error(
                f"Integrated workflow failed: {e}\n{traceback.format_exc()}"
            )
            self._cleanup_temp_files(temp_files_to_cleanup)
            raise ContextualException(
                f"Document processing workflow failed: {str(e)}",
                reason_code=ReasonCode.GCS_WORKFLOW_INTERNAL_ERROR,
                reason_text=str(e)
            )

    # -----------------------------------------------------------------------
    # Config parsing — credentials and rendition codes retrieved here
    # -----------------------------------------------------------------------

    def _parse_processing_config(
        self, activity_request: CustomActivityRequest
    ) -> ProcessingConfig:
        """
        Parse execution attributes and securely retrieve Anthropic credentials.

                Credential source:
                    - Fixed `WORKSPACE_CREDENTIAL_UUID` in this script (primary)
                    - `workspaceCredentialUuid` execution attribute (optional override)
        Optional execution attributes (with defaults):
          - imageDpi                     : image DPI for PDF rendering (default 450)
          - generatePdfOutput            : produce a PDF output (default False)
          - inputRenditionTypeCode       : input rendition type (default SOURCE)
          - inputRenditionSubTypeCode    : input rendition subtype (default ACQUIRED)
          - outputRenditionTypeCode      : output rendition type (default SOURCE)
          - outputRenditionSubTypeCode   : output rendition subtype (default CONVERTED)
        """
        attrs = activity_request.execution_attributes or {}

        # UUID source aligned with current platform guidance:
        # hardcoded activity UUID (with optional execution attribute override).
        credential_uuid = (
            attrs.get("workspaceCredentialUuid")
            or WORKSPACE_CREDENTIAL_UUID
        )

        if not credential_uuid:
            raise ContextualException(
                "Missing credentials UUID configuration",
                reason_code=ReasonCode.REQUIRED_INPUT_NOT_FOUND,
                reason_text=(
                    "Set WORKSPACE_CREDENTIAL_UUID in main.py (or provide "
                    "workspaceCredentialUuid execution attribute override)"
                )
            )

        # Retrieve workspace_id and api_key securely from the credentials microservice
        self._logger.info(f"Retrieving credentials for UUID: {credential_uuid}")
        credential: EncryptedCredential = CredentialsService(
            cars_http_client=self._cars_http_client,
            logger=self._logger
        ).get_encrypted_credential(credential_uuid)

        # Log workspace_id for traceability — NEVER log the api_key
        self._logger.info(
            f"Credential retrieved successfully. "
            f"workspace_id (user_name): {credential.user_name}"
        )

        return ProcessingConfig(
            image_dpi=int(attrs.get("imageDpi", 450)),
            generate_pdf_output=attrs.get("generatePdfOutput", False),
            workspace_id=credential.user_name,
            api_key=credential.password,           # NEVER log this value
            # Rendition codes driven from execution attributes for reusability
            input_rendition_type_code=attrs.get(
                "inputRenditionTypeCode", "SOURCE"
            ),
            input_rendition_sub_type_code=attrs.get(
                "inputRenditionSubTypeCode", "ACQUIRED"
            ),
            output_rendition_type_code=attrs.get(
                "outputRenditionTypeCode", "SOURCE"
            ),
            output_rendition_sub_type_code=attrs.get(
                "outputRenditionSubTypeCode", "CONVERTED"
            )
        )

    # -----------------------------------------------------------------------
    # Input PDF retrieval
    # -----------------------------------------------------------------------

    def _get_input_pdf(self, activity_request, storage_service, config):
        """
        Locate the input PDF rendition GUID from:
          1. activity_request.inputs["acquiredObject"]["renditionGuid"] (preferred)
          2. Execution ledger (fallback) via ExecutionJournalService —
             filtered by configured rendition type/subtype criteria.

        Then download and return the PDF bytes alongside metadata.
        """
        input_rendition_guid = None

        # Attempt 1: read rendition GUID directly from activity inputs
        if activity_request.inputs and 'acquiredObject' in activity_request.inputs:
            acquired_object = activity_request.inputs['acquiredObject']
            input_rendition_guid = acquired_object.get('renditionGuid')
            self._logger.info(
                f"Found rendition GUID in inputs: {input_rendition_guid}"
            )

        # Attempt 2: fall back to execution ledger via shared ExecutionJournalService
        if not input_rendition_guid:
            self._logger.info(
                "No rendition GUID in inputs — checking execution ledger via "
                "ExecutionJournalService..."
            )

            # Use ExecutionJournalService shared library — never call HTTP directly
            journal_service = ExecutionJournalService(
                cars_http_client=self._cars_http_client,
                logger=self._logger
            )
            execution_ledgers_metadata: ExecutionLedgersMetadata = (
                journal_service.get_execution_ledgers_metadata(
                    activity_request.workflow_execution_id
                )
            )
            self._logger.info(
                f"Execution ledger metadata retrieved: {execution_ledgers_metadata}"
            )

            # Use configured rendition type/subtype from execution attributes
            input_rendition_criteria = RenditionCriteria(
                rendition_type_code=config.input_rendition_type_code,
                rendition_sub_type_code=config.input_rendition_sub_type_code
            )

            # Use get_unique_rendition from the shared service — no custom filtering
            input_rendition: LedgerItemsRenditions = (
                journal_service.get_unique_rendition(
                    execution_ledgers_metadata, input_rendition_criteria
                )
            )
            self._logger.info(
                f"Found input rendition from execution ledger: {input_rendition}"
            )

            if input_rendition:
                input_rendition_guid = input_rendition.rendition_guid
                self._logger.info(
                    f"Found rendition GUID in ledger: {input_rendition_guid}"
                )

            # Broaden to any SOURCE rendition if specific match not found
            if not input_rendition_guid:
                self._logger.info(
                    "Specific rendition criteria not matched — broadening to "
                    f"type only: {config.input_rendition_type_code}"
                )
                general_criteria = RenditionCriteria(
                    rendition_type_code=config.input_rendition_type_code
                )
                general_rendition: LedgerItemsRenditions = (
                    journal_service.get_unique_rendition(
                        execution_ledgers_metadata, general_criteria
                    )
                )
                if general_rendition:
                    input_rendition_guid = general_rendition.rendition_guid
                    self._logger.info(
                        f"Found rendition GUID via broadened criteria: "
                        f"{input_rendition_guid}"
                    )

        if not input_rendition_guid:
            raise ContextualException(
                "No input rendition found in inputs or execution ledger",
                reason_code=ReasonCode.GCS_WORKFLOW_INTERNAL_ERROR,
                reason_text="No PDF input rendition available for processing"
            )

        # Retrieve rendition metadata and binary content from storage
        input_meta = storage_service.get_rendition_by_guid(input_rendition_guid)
        self._logger.info(f"Retrieved rendition metadata: {input_meta.file_name}")

        pdf_data = storage_service.get_rendition_data(input_rendition_guid)
        self._logger.info(f"Downloaded PDF data: {len(pdf_data)} bytes")

        # Lightweight wrapper to carry the GUID alongside metadata
        class InputRendition:
            def __init__(self, guid):
                self.rendition_guid = guid

        input_rend = InputRendition(input_rendition_guid)
        return input_rend, input_meta, pdf_data

    # -----------------------------------------------------------------------
    # PDF to HTML conversion (Claude Vision)
    # -----------------------------------------------------------------------

    def _convert_pdf_to_html_image_based(
        self, temp_pdf_path, config, temp_files_to_cleanup
    ):
        """
        Convert PDF to HTML using the image-based Claude Vision converter.
        workspace_id and api_key are passed explicitly — never read from env.
        Platform logger is injected so all output appears in DataDog.
        """
        self._logger.info(
            "Using image-based converter (pdf_processor.py with Claude Vision)..."
        )

        # Pass workspace_id, api_key, and platform logger explicitly
        converter = ImageBasedPDFConverter(
            workspace_id=config.workspace_id,
            api_key=config.api_key,          # NEVER log this value
            logger=self._logger              # inject platform logger for DataDog
        )

        temp_html_path = _create_secure_temp_path(
            prefix="prregs_image_based_", suffix=".html"
        )

        result = converter.convert_pdf_to_html(
            pdf_path=temp_pdf_path,
            output_file=temp_html_path,
            start_page=0,
            end_page=None,
            dpi=config.image_dpi
        )

        if not result.get("success"):
            raise ContextualException(
                "Image-based HTML conversion failed",
                reason_code=ReasonCode.GCS_WORKFLOW_INTERNAL_ERROR,
                reason_text=f"Error: {result.get('error', 'Unknown')}"
            )

        html_file_path = result.get("output_file")
        temp_files_to_cleanup.append(html_file_path)

        with open(html_file_path, 'rb') as f:
            html_bytes = f.read()

        stats = {
            "method": "image-based-claude-vision",
            "pages_processed": result.get("total_pages", 0),
            "successful_pages": result.get("successful_pages", 0),
            "failed_pages": result.get("failed_pages", 0)
        }

        self._logger.info(
            f"Image-based conversion complete: {len(html_bytes)} bytes"
        )
        return html_bytes, html_file_path, stats

    # -----------------------------------------------------------------------
    # Rendition posting helpers
    # -----------------------------------------------------------------------

    def _create_output_rendition(
        self,
        content: str,
        input_rendition,
        input_meta,
        output_crit: RenditionCriteria,
        storage: RenditionStorageService,
        file_extension: str = ".html",
        media_type: str = "text/html"
    ) -> str:
        """
        Post a new rendition to storage derived from the input rendition.

        Args:
            content:          Content string to store.
            input_rendition:  Source rendition object carrying rendition_guid.
            input_meta:       Metadata of the source rendition.
            output_crit:      RenditionCriteria defining type/subtype of the output.
            storage:          RenditionStorageService instance.
            file_extension:   File extension for the output file name.
            media_type:       MIME type of the output content.

        Returns:
            The GUID string of the newly created rendition.
        """
        out_guid = str(uuid.uuid4())
        base = os.path.splitext(input_meta.file_name)[0]
        out_name = f"{base}{file_extension}"

        post = PostRenditionParameters(
            derived_from_rendition_guid=input_rendition.rendition_guid,
            file_name=out_name,
            rendition_guid=out_guid,
            rendition_type_code=output_crit.rendition_type_code,
            rendition_sub_type_code=output_crit.rendition_sub_type_code,
            rendition_sub_type_designator_code=(
                output_crit.rendition_sub_type_designator_code
            ),
            media_type=media_type,
            resource_group_id=input_meta.resource_group_id,
            revision_guid=input_meta.revision_guid,
        )

        storage.post_rendition(post, content.encode("utf-8"))
        self._logger.info(
            f"Created output rendition: {out_guid} "
            f"({output_crit.rendition_type_code}/"
            f"{output_crit.rendition_sub_type_code})"
        )
        return out_guid

    def _post_html_rendition(
        self,
        storage_service,
        input_rend,
        input_meta,
        html_bytes,
        html_file_path,
        config: ProcessingConfig
    ) -> str:
        """
        Post the converted HTML rendition to storage using configured type/subtype.

        Returns:
            GUID of the created HTML rendition.
        """
        # Use rendition codes from execution attributes — not hardcoded
        output_crit = RenditionCriteria(
            rendition_type_code=config.output_rendition_type_code,
            rendition_sub_type_code=config.output_rendition_sub_type_code,
            rendition_sub_type_designator_code=None,
            media_type="text/html"
        )

        # Decode bytes to string if necessary before posting
        html_content = (
            html_bytes.decode('utf-8')
            if isinstance(html_bytes, bytes) else html_bytes
        )

        html_guid = self._create_output_rendition(
            content=html_content,
            input_rendition=input_rend,
            input_meta=input_meta,
            output_crit=output_crit,
            storage=storage_service,
            file_extension=".html",
            media_type="text/html"
        )
        self._logger.info(f"HTML rendition posted successfully: {html_guid}")
        return html_guid

    def _convert_html_to_pdf(
        self,
        html_file_path,
        storage_service,
        input_rend,
        input_meta,
        html_guid,
        temp_files_to_cleanup
    ) -> str:
        """
        Convert the textual HTML rendition to PDF using PyMuPDF Story API,
        then post the result to storage as a SOURCE/CONVERTED rendition.

        Args:
            html_file_path:        Path to the HTML file on disk.
            storage_service:       RenditionStorageService instance.
            input_rend:            Original input rendition object.
            input_meta:            Metadata of the original input rendition.
            html_guid:             GUID of the HTML rendition (used as derivedFrom).
            temp_files_to_cleanup: List to append any new temp files created here.

        Returns:
            GUID of the newly created PDF rendition.
        """
        try:
            converter = HTMLToPDFConverter(logger=self._logger)
            pdf_bytes = converter.convert_html_to_pdf(html_file_path)

            self._logger.info(f"Generated PDF: {len(pdf_bytes)} bytes")

            pdf_guid = str(uuid.uuid4())
            base = os.path.splitext(input_meta.file_name)[0]
            pdf_name = f"{base}_converted.pdf"

            # Build PostRenditionParameters for the PDF output rendition
            pdf_post_params = PostRenditionParameters(
                derived_from_rendition_guid=html_guid,
                file_name=pdf_name,
                rendition_guid=pdf_guid,
                rendition_type_code="SOURCE",
                rendition_sub_type_code="CONVERTED",
                rendition_sub_type_designator_code=None,
                media_type="application/pdf",
                resource_group_id=input_meta.resource_group_id,
                revision_guid=input_meta.revision_guid,
            )

            storage_service.post_rendition(pdf_post_params, pdf_bytes)
            self._logger.info(f"PDF rendition created and posted: {pdf_guid}")
            return pdf_guid

        except Exception as e:
            self._logger.error(f"HTML → PDF conversion failed: {e}")
            raise ContextualException(
                f"HTML → PDF conversion failed: {str(e)}",
                reason_code=ReasonCode.GCS_WORKFLOW_INTERNAL_ERROR,
                reason_text=str(e)
            )

    # -----------------------------------------------------------------------
    # Temporary file cleanup
    # -----------------------------------------------------------------------

    def _cleanup_temp_files(self, temp_files_to_cleanup: List[str]) -> None:
        """
        Securely delete all temporary files created during processing.
        Failures are logged as warnings but do not raise exceptions.

        Args:
            temp_files_to_cleanup: List of absolute file paths to delete.
        """
        for temp_file in temp_files_to_cleanup:
            if temp_file and os.path.exists(temp_file):
                try:
                    _secure_delete_file(temp_file)
                    self._logger.info(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    # Log warning but do not raise — cleanup failures
                    # should not block the activity result
                    self._logger.warn(f"Failed to cleanup {temp_file}: {e}")

    # -----------------------------------------------------------------------
    # Success response builder
    # -----------------------------------------------------------------------

    def _create_success_response(
        self,
        html_guid: str,
        pdf_guid: Optional[str],
        conversion_stats: dict,
        config: ProcessingConfig
    ) -> CustomActivityResponse:
        """
        Build the CustomActivityResponse for a successful run.

        Args:
            html_guid:         GUID of the created HTML rendition.
            pdf_guid:          GUID of the created PDF rendition, or None if skipped.
            conversion_stats:  Stats dict from the image-based conversion step.
            config:            ProcessingConfig used during this run.

        Returns:
            CustomActivityResponse with SUCCEEDED result code and output_attributes.
        """
        # Build base output attributes with HTML rendition details
        output_attrs = {
            "htmlRendition": {
                "renditionGuid": html_guid,
                "conversionMethod": "image-based-enhanced-v2 (Claude Vision)",
                "pagesProcessed": conversion_stats.get("pages_processed", 0),
                "successfulPages": conversion_stats.get("successful_pages", 0),
                "failedPages": conversion_stats.get("failed_pages", 0)
            },
            "processingConfig": {
                "imageDpi": config.image_dpi,
                "generatePdfOutput": config.generate_pdf_output,
                "htmlToPdfAvailable": HTML_TO_PDF_AVAILABLE,
                "htmlToPdfMethod": "PyMuPDF Story API from textual HTML",
                "enhancements": (
                    "Refined AI prompt, robust HTML cleanup, "
                    "corrected PDF rendering logic"
                )
            }
        }

        # Include PDF rendition details if PDF output was requested
        if pdf_guid:
            # PDF was successfully generated — include its GUID
            output_attrs["pdfRendition"] = {
                "renditionGuid": pdf_guid,
                "derivedFrom": html_guid,
                "status": "success"
            }
        elif config.generate_pdf_output:
            # PDF was requested but was not produced — surface this in output
            output_attrs["pdfRendition"] = {
                "status": "failed",
                "reason": "PDF generation was requested but failed"
            }

        # Return the final success response with all output attributes
        return CustomActivityResponse(
            result_code=WorkflowResultCode.SUCCEEDED,
            output_attributes=output_attrs
        )
