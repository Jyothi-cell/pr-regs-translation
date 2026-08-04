# pdf_processor.py
# Activity-scoped helper for image-based PDF → HTML conversion using Claude Vision API.
# Credentials are passed explicitly from main.py — never read from environment variables.
# All logging uses the injected GcsLogger for DataDog visibility.

import re
import os
import base64
import time
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image
from gcs_jsonlogger.gcs_logger import GcsLogger

# ---------------------------------------------------------------------------
# GCS AI Platform client — optional, preferred over raw API key
# ---------------------------------------------------------------------------
try:
    from anthropic_client import get_anthropic_credentials, create_anthropic_client
    USE_GCS_AI_PLATFORM = True
except ImportError:
    USE_GCS_AI_PLATFORM = False
    import anthropic  # fallback: direct Anthropic SDK


# ---------------------------------------------------------------------------
# Configuration — sourced from environment for local dev only.
# In platform runtime, credentials come exclusively from CredentialsService.
# ---------------------------------------------------------------------------
CLAUDE_MODEL = os.environ.get("MODEL_NAME", "claude-sonnet-4-5")
MAX_PDF_SIZE_MB = float(os.environ.get("MAX_PDF_SIZE_MB", "50"))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "1000"))
MIN_API_CALL_INTERVAL_SECONDS = float(os.environ.get("MIN_API_CALL_INTERVAL_SECONDS", "0.5"))


# ============================================================================
# IMAGE-BASED PDF TO HTML CONVERTER
# ============================================================================

class ImageBasedPDFConverter:
    """
    Convert PDF pages to HTML using image-based OCR with Claude Vision API.

    Credentials (workspace_id, api_key) are passed explicitly from main.py
    via the GCS Credentials microservice — never read from environment variables.
    """

    def __init__(
        self,
        workspace_id: str = None,
        api_key: str = None,           # Matches main.py kwarg — NEVER log this value
        claude_api_key: str = None,    # Legacy alias for api_key
        model: str = CLAUDE_MODEL,
        logger: GcsLogger = None
    ):
        """
        Initialize the converter.

        Args:
            workspace_id:   GCS AI Platform workspace ID (from CredentialsService).
            api_key:        Anthropic API key (from CredentialsService — NEVER log).
            claude_api_key: Legacy alias for api_key (for backward compatibility).
            model:          Claude model name.
            logger:         GcsLogger instance injected from main.py for DataDog logging.
        """
        # Use injected logger; fall back to module-level GcsLogger if not provided
        self._logger = logger or GcsLogger(__name__)

        self.model = model
        self.claude_client = None

        # Normalize api_key — accept either kwarg name for compatibility
        resolved_key = api_key or claude_api_key

        # Attempt GCS AI Platform client first (preferred in platform runtime).
        # This refreshes temporary Anthropic keys on each run. If it fails,
        # fall back to the credential.password value passed from main.py.
        if USE_GCS_AI_PLATFORM and workspace_id:
            try:
                self._logger.info("Attempting to initialize Claude client via GCS AI Platform...")
                credentials = get_anthropic_credentials(
                    workspace_id=workspace_id,
                    api_key=resolved_key,
                    model_name=model
                )
                self.claude_client = create_anthropic_client(credentials["api_key"])
                self.model = credentials.get("model_name", model)
                self._logger.info(
                    f"Claude client initialized via GCS AI Platform (model: {self.model})"
                )
            except Exception as e:
                self._logger.warn(
                    f"GCS AI Platform credential retrieval failed: {e}. "
                    f"Falling back to explicit API key."
                )

        # Fallback: initialize Anthropic client directly with the resolved API key
        if not self.claude_client:
            if not resolved_key:
                raise ValueError(
                    "No Claude API key provided. "
                    "Pass api_key from the GCS Credentials microservice."
                )
            # Import here to avoid top-level import when USE_GCS_AI_PLATFORM is True
            import anthropic as _anthropic
            self.claude_client = _anthropic.Anthropic(api_key=resolved_key)
            self._logger.info(f"Claude client initialized with explicit API key (model: {self.model})")

        # PDF state
        self.pdf_document = None
        self.total_pages = 0

        # Limits
        self.max_pdf_size_bytes = int(MAX_PDF_SIZE_MB * 1024 * 1024)
        self.max_pdf_pages = MAX_PDF_PAGES
        self.min_api_call_interval_seconds = MIN_API_CALL_INTERVAL_SECONDS

        # Rate limiting state
        self._last_api_call_at: float = 0.0

        # Cache the vision prompt — built once, reused per page
        self._vision_prompt: str = self._create_vision_prompt()

    # -----------------------------------------------------------------------
    # PDF Loading
    # -----------------------------------------------------------------------

    def load_pdf(self, pdf_path: str = None, pdf_bytes: bytes = None) -> bool:
        """
        Load PDF from file path or bytes.

        Args:
            pdf_path:  Path to PDF file on disk.
            pdf_bytes: PDF content as bytes.

        Returns:
            True on success, False on failure.
        """
        try:
            if pdf_path:
                file_size = os.path.getsize(pdf_path)
                if file_size > self.max_pdf_size_bytes:
                    raise ValueError(
                        f"PDF size {file_size} bytes exceeds limit of "
                        f"{self.max_pdf_size_bytes} bytes ({MAX_PDF_SIZE_MB} MB)"
                    )
                self.pdf_document = fitz.open(pdf_path)
                self._logger.info(f"Loaded PDF from path: {pdf_path}")

            elif pdf_bytes:
                if len(pdf_bytes) > self.max_pdf_size_bytes:
                    raise ValueError(
                        f"PDF size {len(pdf_bytes)} bytes exceeds limit of "
                        f"{self.max_pdf_size_bytes} bytes ({MAX_PDF_SIZE_MB} MB)"
                    )
                self.pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                self._logger.info("Loaded PDF from bytes")

            else:
                raise ValueError("Either pdf_path or pdf_bytes must be provided")

            self.total_pages = len(self.pdf_document)

            # Enforce page count limit
            if self.total_pages > self.max_pdf_pages:
                try:
                    self.pdf_document.close()
                except Exception:
                    pass
                self.pdf_document = None
                raise ValueError(
                    f"PDF page count {self.total_pages} exceeds limit of "
                    f"{self.max_pdf_pages} pages"
                )

            self._logger.info(f"Total pages in PDF: {self.total_pages}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to load PDF: {e}\n{traceback.format_exc()}")
            return False

    # -----------------------------------------------------------------------
    # Image Extraction
    # -----------------------------------------------------------------------

    def extract_page_as_image(self, page_num: int, dpi: int = 450) -> Optional[bytes]:
        """
        Extract a single PDF page as a PNG image.

        Args:
            page_num: Page number (0-indexed).
            dpi:      Rendering resolution (default 450 for high quality).

        Returns:
            PNG image bytes, or None on failure.
        """
        try:
            page = self.pdf_document[page_num]
            zoom = dpi / 72  # PyMuPDF baseline is 72 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            return pix.tobytes("png")
        except Exception as e:
            self._logger.error(
                f"Error extracting page {page_num + 1} as image: {e}\n"
                f"{traceback.format_exc()}"
            )
            return None

    # -----------------------------------------------------------------------
    # Image Compression
    # -----------------------------------------------------------------------

    def compress_image_if_needed(
        self,
        image_bytes: bytes,
        max_size_mb: float = 4.5,
        quality: int = 85
    ) -> tuple:
        """
        Compress image if it exceeds the Claude Vision 5MB API limit.

        Always returns a consistent tuple: (image_bytes, media_type).

        Args:
            image_bytes: Original PNG image bytes.
            max_size_mb: Maximum allowed size in MB (default 4.5 to buffer below 5MB).
            quality:     JPEG quality for compression (1–100).

        Returns:
            Tuple of (bytes, media_type_str).
        """
        max_size_bytes = int(max_size_mb * 1024 * 1024)
        original_size = len(image_bytes)

        # Under limit — return original PNG unchanged
        if original_size <= max_size_bytes:
            return image_bytes, 'image/png'

        self._logger.info(
            f"Image size {original_size / (1024 * 1024):.2f}MB exceeds "
            f"{max_size_mb}MB limit — compressing..."
        )

        try:
            img = Image.open(BytesIO(image_bytes))

            # Flatten transparency to white background for JPEG compatibility
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                mask = img.split()[-1] if img.mode in ('RGBA', 'LA') else None
                background.paste(img, mask=mask)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Iterative quality reduction
            current_quality = quality
            output = BytesIO()
            img.save(output, format='JPEG', quality=current_quality, optimize=True)
            compressed_bytes = output.getvalue()

            while len(compressed_bytes) > max_size_bytes and current_quality > 50:
                current_quality -= 10
                output = BytesIO()
                img.save(output, format='JPEG', quality=current_quality, optimize=True)
                compressed_bytes = output.getvalue()

            # Iterative resize if quality reduction is insufficient
            if len(compressed_bytes) > max_size_bytes:
                self._logger.warn("Quality reduction insufficient — resizing image...")
                scale_factor = 0.8
                while len(compressed_bytes) > max_size_bytes and scale_factor > 0.3:
                    new_width = int(img.width * scale_factor)
                    new_height = int(img.height * scale_factor)
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    output = BytesIO()
                    resized_img.save(output, format='JPEG', quality=70, optimize=True)
                    compressed_bytes = output.getvalue()
                    scale_factor -= 0.1

            self._logger.info(
                f"Compressed image from {original_size / (1024 * 1024):.2f}MB "
                f"to {len(compressed_bytes) / (1024 * 1024):.2f}MB"
            )
            return compressed_bytes, 'image/jpeg'

        except Exception as e:
            self._logger.error(
                f"Image compression failed: {e}\n{traceback.format_exc()} "
                f"— using original PNG"
            )
            return image_bytes, 'image/png'

    # -----------------------------------------------------------------------
    # Vision Prompt (cached at construction time)
    # -----------------------------------------------------------------------

    def _create_vision_prompt(self) -> str:
        """
        Build the Claude Vision prompt for PDF page → HTML conversion.
        Called once in __init__ and cached as self._vision_prompt.
        """
        return """You are tasked with translating and formatting the content of a PDF page image. Follow these instructions carefully:

You will be provided with an image of a PDF page.

MANDATORY TRANSLATION TO ENGLISH: Translate ALL content from the source language to ENGLISH. This is a critical requirement:

Translate each and every word, phrase, heading, table content, list item, and footnote to English
Do NOT leave any text in the original language - everything must be translated to English
Maintain the exact meaning and nuance of the original text in the English translation
If you encounter text that is already in English, keep it as-is
Do not skip any text - translate everything completely to English

🔴 CRITICAL RULE #1: FORMATTING MUST MATCH SOURCE EXACTLY 100%
Before you begin, understand this absolute requirement:
- Your ONLY job is to translate text and preserve EXISTING formatting
- You are FORBIDDEN from adding, removing, or changing ANY formatting
- Look at the source image with extreme care before applying ANY formatting tag
- When in doubt about formatting, ALWAYS choose plain text
- Formatting accuracy is MORE IMPORTANT than assumptions about what "should" be formatted

🔴 CRITICAL RULE #2: FORBIDDEN FORMATTING TYPES
These formatting types are ABSOLUTELY FORBIDDEN - NEVER use them:
- ❌ STRIKETHROUGH: Never use <del>, <s>, <strike>, or text-decoration: line-through
- ❌ UNDERLINE: Never use <u> or text-decoration: underline (unless you see actual underlined text)
- ❌ BACKGROUND COLORS: Never use background-color or highlighting in HTML
- ❌ TEXT DECORATIONS: Ignore PDF annotations, highlights, markups - they are NOT text formatting

🔴 CRITICAL RULE #3: PDF ANNOTATIONS ARE NOT TEXT FORMATTING
- Yellow highlights = PDF annotation = IGNORE completely
- Colored backgrounds = PDF annotation = IGNORE completely
- Markup tools from PDF viewers = NOT part of the text = IGNORE completely
- Only format text based on the actual text appearance (bold, italic), NOT on annotations

After translation to English, format the text in HTML. Strive to maintain the format as close to the original image as possible. Pay special attention to:

Lists (bulleted, numbered, alphabetical, roman numerals, or with hyphens)
Tables (translate ALL table content to English)
Indentation
Text alignment
Font styles (bold, italic ONLY - and ONLY if visibly present in source)
Font sizes (use appropriate HTML tags to represent different sizes)

🔴 ABSOLUTE FORMATTING RULES - EXACT SOURCE MATCHING ONLY:

RULE 1: BOLD TEXT - VISUAL VERIFICATION MANDATORY
Apply <strong> or <b> tags ONLY when:
  ✓ The text is VISIBLY THICKER/HEAVIER than surrounding regular text
  ✓ You can clearly see the text is darker/bolder than normal body text
  ✓ Another person looking at the image would agree it's bold
  ✗ Do NOT assume section numbers, headings, or list markers are bold
  ✗ Do NOT make text bold because it "seems important"
  ✗ Do NOT make text bold because of its position or role

RULE 2: ITALIC TEXT - VISUAL VERIFICATION MANDATORY
Apply <em> or <i> tags ONLY when:
  ✓ The text is VISIBLY SLANTED compared to regular upright text
  ✓ You can clearly see the text angle is different from normal text
  ✗ Do NOT assume case citations or legal references are italic

RULE 3: LIST MARKERS ARE USUALLY PLAIN TEXT
DEFAULT: List markers (A., B., C., 1., 2., 3., a., b., i., ii.) are PLAIN TEXT in 99% of documents

RULE 4: WHEN UNCERTAIN, CHOOSE PLAIN TEXT
If you have ANY doubt about whether text is bold or italic → Use plain text with NO formatting tags

RULE 5: TRANSLATION ONLY CHANGES LANGUAGE
Any language → English: Format in source = Format in output (no additions, no removals)

🔴 MANDATORY PRE-OUTPUT VERIFICATION PROCESS:
1. SCAN every <strong>, <b>, <em>, <i> tag in your HTML
2. For EACH tag, look back at the source image at that exact text
3. VERIFY the formatting is clearly visible in the source
4. REMOVE any tag where you cannot clearly confirm the formatting
5. CHECK: Did I add ANY strikethrough? If YES, REMOVE IT IMMEDIATELY
6. CHECK: Did I add ANY underline where text isn't underlined? If YES, REMOVE IT

For tables: Use proper HTML table tags. Translate ALL table content to English.
For lists: Preserve EXACT list structure. ALL CONSECUTIVE LIST ITEMS MUST BE IN THE SAME LIST STRUCTURE.
For indentation: Use inline CSS styles (e.g., style="text-indent: 20px;")

Your output should be in HTML format only with ALL content translated to ENGLISH. Do not include any additional tags, explanations, or extra information. The response should contain pure HTML code ready to be rendered."""

    # -----------------------------------------------------------------------
    # Formatting Post-Processor
    # -----------------------------------------------------------------------

    def _clean_formatting_issues(self, html_content: str) -> str:
        """
        Post-process HTML to remove incorrect bold/italic formatting that
        Claude may have incorrectly applied to list markers and section numbers.

        Args:
            html_content: Raw HTML string from Claude Vision response.

        Returns:
            Cleaned HTML string.
        """
        patterns = [
            # Uppercase letter list markers: <strong>A.</strong>
            (r'<(?:strong|b)>([A-Z]\.)</(?:strong|b)>(?=\s)', r'\1'),
            # Lowercase letter list markers: <strong>a.</strong>
            (r'<(?:strong|b)>([a-z]\.)</(?:strong|b)>(?=\s)', r'\1'),
            # Number list markers: <strong>1.</strong>
            (r'<(?:strong|b)>(\d+\.)</(?:strong|b)>(?=\s)', r'\1'),
            # Parenthesized letters: <strong>(a)</strong>
            (r'<(?:strong|b)>\(([a-zA-Z])\)</(?:strong|b)>(?=\s)', r'(\1)'),
            # Parenthesized numbers: <strong>(1)</strong>
            (r'<(?:strong|b)>\((\d+)\)</(?:strong|b)>(?=\s)', r'(\1)'),
            # Roman numerals: <strong>(i)</strong>
            (r'<(?:strong|b)>\(([ivxlcdm]+)\)</(?:strong|b)>(?=\s)', r'(\1)'),
            # Letter with closing paren: <strong>a)</strong>
            (r'<(?:strong|b)>([a-z]\))</(?:strong|b)>(?=\s)', r'\1'),
            # Number with closing paren: <strong>1)</strong>
            (r'<(?:strong|b)>(\d+\))</(?:strong|b)>(?=\s)', r'\1'),
            # Section/Article markers: <strong>Article 5.</strong>
            (r'<(?:strong|b)>((?:Article|Section|Chapter)\s+\d+\.?)</(?:strong|b)>', r'\1'),
        ]

        cleaned = html_content
        for pattern, replacement in patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        return cleaned

    # -----------------------------------------------------------------------
    # Per-Page Processing
    # -----------------------------------------------------------------------

    def process_page_to_html(
        self,
        page_num: int,
        dpi: int = 450,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Process a single PDF page to HTML using Claude Vision API with retry logic.

        Args:
            page_num:    Page number (0-indexed).
            dpi:         Resolution for image extraction.
            max_retries: Maximum retry attempts for incomplete/truncated responses.

        Returns:
            Dict with keys: page_num, status, html_content, content_length, attempts.
        """
        self._logger.info(f"Processing page {page_num + 1}/{self.total_pages}...")

        for attempt in range(max_retries + 1):
            try:
                # Extract page as PNG image
                image_bytes = self.extract_page_as_image(page_num, dpi)
                if not image_bytes:
                    return {
                        "page_num": page_num + 1,
                        "status": "failed",
                        "error": "Failed to extract page image",
                        "html_content": "",
                        "attempts": attempt + 1
                    }

                # Compress if needed — always returns a consistent (bytes, media_type) tuple
                image_bytes, media_type = self.compress_image_if_needed(image_bytes)

                # Base64-encode for Claude Vision API
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                # Rate limiting — enforce minimum interval between API calls
                now = time.time()
                elapsed = now - self._last_api_call_at
                if elapsed < self.min_api_call_interval_seconds:
                    sleep_for = self.min_api_call_interval_seconds - elapsed
                    time.sleep(sleep_for)

                # Call Claude Vision API
                response = self.claude_client.messages.create(
                    model=self.model,
                    max_tokens=32000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_base64
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": self._vision_prompt  # Use cached prompt
                                }
                            ]
                        }
                    ]
                )
                self._last_api_call_at = time.time()

                # Extract raw HTML from response
                html_content = response.content[0].text.strip()

                # Handle truncation — retry if max_tokens was hit
                if response.stop_reason == "max_tokens":
                    self._logger.warn(
                        f"Page {page_num + 1} response truncated (max_tokens reached)"
                    )
                    if attempt < max_retries:
                        self._logger.info(
                            f"Retrying page {page_num + 1} "
                            f"(attempt {attempt + 1}/{max_retries})..."
                        )
                        continue
                    else:
                        self._logger.warn(
                            f"Page {page_num + 1} still truncated after "
                            f"{max_retries} retries — proceeding with partial content"
                        )

                # Strip markdown code fences if Claude wrapped output
                html_content = html_content.replace("```html", "").replace("```", "").strip()

                # Apply formatting post-processor to remove false-positive bold/italic
                html_content = self._clean_formatting_issues(html_content)

                # Validate response length — retry if suspiciously short
                if len(html_content) < 100:
                    if attempt < max_retries:
                        self._logger.warn(
                            f"Page {page_num + 1} response too short "
                            f"({len(html_content)} chars) — retrying "
                            f"(attempt {attempt + 1}/{max_retries})..."
                        )
                        continue
                    else:
                        self._logger.warn(
                            f"Page {page_num + 1} response still short after "
                            f"{max_retries} retries"
                        )

                # Validate HTML structural completeness — retry if truncated mid-tag
                closing_tags = [
                    '</html>', '</body>', '</div>', '</p>',
                    '</table>', '</ul>', '</ol>', '</li>'
                ]
                is_structurally_complete = any(
                    html_content.strip().endswith(tag) for tag in closing_tags
                )
                if not is_structurally_complete:
                    if attempt < max_retries:
                        self._logger.warn(
                            f"Page {page_num + 1} HTML appears structurally incomplete "
                            f"— retrying (attempt {attempt + 1}/{max_retries})..."
                        )
                        continue
                    else:
                        self._logger.warn(
                            f"Page {page_num + 1} HTML still appears incomplete "
                            f"after {max_retries} retries — proceeding"
                        )

                self._logger.info(
                    f"Page {page_num + 1} processed successfully "
                    f"({len(html_content)} chars, attempt {attempt + 1})"
                )

                return {
                    "page_num": page_num + 1,
                    "status": "success",
                    "html_content": html_content,
                    "content_length": len(html_content),
                    "attempts": attempt + 1
                }

            except Exception as e:
                self._logger.error(
                    f"Page {page_num + 1} attempt {attempt + 1} failed: {e}\n"
                    f"{traceback.format_exc()}"
                )
                if attempt < max_retries:
                    self._logger.info(
                        f"Retrying page {page_num + 1} "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    continue

        # Exhausted all retries — return structured failure
        return {
            "page_num": page_num + 1,
            "status": "failed",
            "error": "Max retries exceeded",
            "html_content": (
                f"<!-- Error processing page {page_num + 1}: Max retries exceeded -->"
            ),
            "attempts": max_retries + 1
        }

    # -----------------------------------------------------------------------
    # All-Pages Processing
    # -----------------------------------------------------------------------

    def process_all_pages(
        self,
        start_page: int = 0,
        end_page: int = None,
        dpi: int = 450
    ) -> List[Dict[str, Any]]:
        """
        Process a range of PDF pages to HTML.

        Args:
            start_page: Starting page number (0-indexed).
            end_page:   Ending page number (0-indexed, None = all pages).
            dpi:        Image resolution for extraction.

        Returns:
            List of per-page result dicts.
        """
        if not self.pdf_document:
            raise ValueError("PDF not loaded. Call load_pdf() first.")

        if end_page is None:
            end_page = self.total_pages - 1

        results = []
        for page_num in range(start_page, end_page + 1):
            result = self.process_page_to_html(page_num, dpi)
            results.append(result)

        return results

    # -----------------------------------------------------------------------
    # HTML Document Assembly
    # -----------------------------------------------------------------------

    def create_complete_html_document(self, page_results: List[Dict[str, Any]]) -> str:
        """
        Assemble a complete HTML document from per-page conversion results.

        Args:
            page_results: List of dicts returned by process_all_pages().

        Returns:
            Complete HTML document string.
        """
        pages_html = []
        for result in page_results:
            if result['status'] == 'success':
                pages_html.append(f"""
    <div class="page" data-page="{result['page_num']}">
        <div class="page-header">Page {result['page_num']}</div>
        {result['html_content']}
    </div>
""")
            else:
                pages_html.append(f"""
    <div class="page error" data-page="{result['page_num']}">
        <div class="page-header">Page {result['page_num']} - Error</div>
        <p class="error-message">{result.get('error', 'Unknown error')}</p>
    </div>
""")

        combined_content = '\n'.join(pages_html)

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Document - HTML Conversion</title>
    <style>
        body {{
            font-family: "Calibri", Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .document-header {{
            text-align: center;
            border-bottom: 2px solid #2F5496;
            padding-bottom: 20px;
            margin-bottom: 30px;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
        }}
        .page {{
            background-color: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            page-break-after: always;
        }}
        .page-header {{
            font-size: 10pt;
            color: #666;
            text-align: right;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .page.error {{
            background-color: #fff3cd;
            border: 1px solid #ffc107;
        }}
        .error-message {{
            color: #856404;
            font-style: italic;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 10pt 0;
            font-size: 10pt;
        }}
        th, td {{
            border: 1px solid #333;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background-color: #f0f5ff;
            font-weight: bold;
        }}
        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}
        li {{
            margin: 5px 0;
        }}
        ol[type="a"], ol[type="A"] {{
            list-style-type: lower-alpha;
        }}
        @media print {{
            body {{ background-color: white; }}
            .page {{ box-shadow: none; margin-bottom: 0; }}
        }}
    </style>
</head>
<body>
    <div class="document-content">
{combined_content}
    </div>
</body>
</html>'''

    # -----------------------------------------------------------------------
    # Main Conversion Entry Point
    # -----------------------------------------------------------------------

    def convert_pdf_to_html(
        self,
        pdf_path: str = None,
        pdf_bytes: bytes = None,
        output_file: str = None,
        start_page: int = 0,
        end_page: int = None,
        dpi: int = 450
    ) -> Dict[str, Any]:
        """
        Complete PDF to HTML conversion workflow.
        Called by main.py CustomActivityRunner.

        Args:
            pdf_path:    Path to PDF file on disk.
            pdf_bytes:   PDF content as bytes.
            output_file: Output HTML file path (required by main.py).
            start_page:  Starting page (0-indexed).
            end_page:    Ending page (0-indexed, None = all pages).
            dpi:         Image resolution DPI (default 450).

        Returns:
            Dict with keys: success, output_file, total_pages,
                            successful_pages, failed_pages, page_results.
        """
        try:
            # Load the PDF from path or bytes
            if not self.load_pdf(pdf_path, pdf_bytes):
                return {"success": False, "error": "Failed to load PDF"}

            # FIX: Use explicit None check — end_page=0 is falsy but valid
            end_display = (
                (end_page + 1) if end_page is not None else self.total_pages
            )
            self._logger.info(
                f"Processing pages {start_page + 1} to {end_display} "
                f"at {dpi} DPI..."
            )

            # Process all pages in the specified range
            page_results = self.process_all_pages(start_page, end_page, dpi)

            # Assemble the full HTML document from per-page results
            html_content = self.create_complete_html_document(page_results)

            # Generate output filename if not provided by caller
            if not output_file:
                if pdf_path:
                    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                else:
                    base_name = "document"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"{base_name}_vision_{timestamp}.html"

            # Write the assembled HTML to disk
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Calculate success/failure statistics
            successful = sum(
                1 for r in page_results if r['status'] == 'success'
            )
            failed = len(page_results) - successful

            self._logger.info(
                f"Conversion complete. Output: {output_file} | "
                f"Pages: {len(page_results)} | "
                f"Successful: {successful} | Failed: {failed}"
            )

            # Return contract expected by main.py CustomActivityRunner
            return {
                "success": True,
                "output_file": output_file,
                "total_pages": len(page_results),
                "successful_pages": successful,
                "failed_pages": failed,
                "page_results": page_results
            }

        except Exception as e:
            import traceback
            self._logger.error(
                f"Conversion failed: {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": str(e)}

        finally:
            # Always release the PyMuPDF document handle
            if self.pdf_document:
                try:
                    self.pdf_document.close()
                except Exception:
                    pass
                self.pdf_document = None
