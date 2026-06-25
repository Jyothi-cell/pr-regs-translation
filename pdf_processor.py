import anthropic    
import json    
import os    
import base64    
from typing import Dict, List, Any, Tuple    
from datetime import datetime    
from io import BytesIO    
import fitz  # PyMuPDF
from PIL import Image    
try:    
    from dotenv import load_dotenv    
    load_dotenv()    
except ImportError:    
    # dotenv not available in production, environment variables already set    
    pass  
    
# ============================================================================    
# CONFIGURATION    
# ============================================================================    
CLAUDE_MODEL = "claude-sonnet-4-20250514"    
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")    
BASE_DATA_PATH = f"{os.environ.get('BASE_PATH', os.path.curdir)}/codes_reimagine_platform"    
APP_DATA_PATH = f"{BASE_DATA_PATH}/legislation_data/MI_Pending_Doc"
    
# GCS AI Platform support    
try:    
    from anthropic_client import get_anthropic_credentials, create_anthropic_client    
    USE_GCS_AI_PLATFORM = True    
except ImportError:    
    USE_GCS_AI_PLATFORM = False    
    print("  Note: anthropic_client not available, using environment API key")
    
# ============================================================================    
# IMAGE-BASED PDF TO HTML CONVERTER    
# ============================================================================
    
class ImageBasedPDFConverter:    
    """Convert PDF pages to HTML using image-based OCR with Claude Vision"""
        
    def __init__(self, claude_api_key: str = None, model: str = CLAUDE_MODEL, workspace_id: str = None):    
        """    
        Initialize the converter.
            
        Args:    
            claude_api_key: Optional API key    
            model: Claude model to use    
            workspace_id: GCS workspace ID    
        """    
        self.claude_client = None    
        self.model = model
            
        # Try GCS AI Platform first    
        if not claude_api_key and USE_GCS_AI_PLATFORM:    
            try:    
                print("  Attempting to get credentials from GCS AI Platform...")    
                credentials = get_anthropic_credentials(workspace_id=workspace_id, model_name=model)    
                self.claude_client = create_anthropic_client(credentials)    
                self.model = credentials.get("model_name", model)    
                print(f"✓ Claude client initialized via GCS AI Platform (model: {self.model})")    
            except Exception as e:    
                print(f"  Warning: Failed to get credentials from GCS AI Platform: {e}")    
                claude_api_key = CLAUDE_API_KEY
            
        # Fallback to API key    
        if not self.claude_client:    
            if not claude_api_key:    
                claude_api_key = CLAUDE_API_KEY
                
            if not claude_api_key:    
                raise ValueError("No Claude API key provided or found in environment")
                
            self.claude_client = anthropic.Anthropic(api_key=claude_api_key)    
            print(f"✓ Claude client initialized successfully (model: {self.model})")
            
        self.pdf_document = None    
        self.total_pages = 0
        
    def load_pdf(self, pdf_path: str = None, pdf_bytes: bytes = None) -> bool:    
        """    
        Load PDF from file path or bytes.
            
        Args:    
            pdf_path: Path to PDF file    
            pdf_bytes: PDF content as bytes
            
        Returns:    
            bool: Success status    
        """    
        try:    
            if pdf_path:    
                self.pdf_document = fitz.open(pdf_path)    
                print(f"✓ Loaded PDF from path: {pdf_path}")    
            elif pdf_bytes:    
                self.pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")    
                print(f"✓ Loaded PDF from bytes")    
            else:    
                raise ValueError("Either pdf_path or pdf_bytes must be provided")
                
            self.total_pages = len(self.pdf_document)    
            print(f"  Total pages: {self.total_pages}")    
            return True
                
        except Exception as e:    
            print(f"✗ Failed to load PDF: {e}")    
            return False
        
    def extract_page_as_image(self, page_num: int, dpi: int = 450) -> bytes:  # INCREASED DEFAULT DPI  
        """    
        Extract a single page as PNG image.
            
        Args:    
            page_num: Page number (0-indexed)    
            dpi: Resolution for image extraction (increased to 450 for better quality)
            
        Returns:    
            bytes: PNG image data    
        """    
        try:    
            page = self.pdf_document[page_num]    
            # Higher DPI for better quality    
            zoom = dpi / 72  # 72 is default DPI    
            mat = fitz.Matrix(zoom, zoom)    
            pix = page.get_pixmap(matrix=mat)    
            return pix.tobytes("png")    
        except Exception as e:    
            print(f"  Error extracting page {page_num + 1} as image: {e}")    
            return None
    
    def compress_image_if_needed(self, image_bytes: bytes, max_size_mb: float = 4.5, quality: int = 85) -> bytes:
        """
        Compress image if it exceeds maximum size limit.
        
        Args:
            image_bytes: Original PNG image bytes
            max_size_mb: Maximum allowed size in MB (default 4.5MB to leave buffer below 5MB limit)
            quality: JPEG quality for compression (1-100, default 85)
            
        Returns:
            bytes: Compressed image bytes (JPEG format if compressed, original PNG if small enough)
        """
        max_size_bytes = int(max_size_mb * 1024 * 1024)
        original_size = len(image_bytes)
        
        # If under limit, return original
        if original_size <= max_size_bytes:
            return image_bytes
        
        print(f"    Image size {original_size / (1024*1024):.2f}MB exceeds {max_size_mb}MB limit, compressing...")
        
        try:
            # Load image using PIL
            img = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if necessary (PNG may have alpha channel)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Try compression at specified quality first
            output = BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            compressed_bytes = output.getvalue()
            compressed_size = len(compressed_bytes)
            
            # If still too large, reduce quality iteratively
            current_quality = quality
            while compressed_size > max_size_bytes and current_quality > 50:
                current_quality -= 10
                output = BytesIO()
                img.save(output, format='JPEG', quality=current_quality, optimize=True)
                compressed_bytes = output.getvalue()
                compressed_size = len(compressed_bytes)
            
            # If still too large, resize the image
            if compressed_size > max_size_bytes:
                print(f"    Quality reduction insufficient, resizing image...")
                scale_factor = 0.8
                while compressed_size > max_size_bytes and scale_factor > 0.3:
                    new_width = int(img.width * scale_factor)
                    new_height = int(img.height * scale_factor)
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    output = BytesIO()
                    resized_img.save(output, format='JPEG', quality=70, optimize=True)
                    compressed_bytes = output.getvalue()
                    compressed_size = len(compressed_bytes)
                    scale_factor -= 0.1
            
            print(f"    ✓ Compressed from {original_size / (1024*1024):.2f}MB to {compressed_size / (1024*1024):.2f}MB")
            
            # Return JPEG bytes with proper media type handling
            return compressed_bytes, 'image/jpeg'
            
        except Exception as e:
            print(f"    ✗ Compression failed: {e}, using original image")
            return image_bytes, 'image/png'
        
    def create_vision_prompt(self) -> str:    
        """Create the vision prompt for Claude - ENHANCED VERSION"""    
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
VISUAL TEST: Compare text thickness to regular body text. If you cannot see a clear difference, use plain text.
EXAMPLE: If "Article 10." looks the same thickness as body text → Article 10. (plain)
EXAMPLE: If "Article 10." is clearly thicker than body text → <strong>Article 10.</strong>

RULE 2: ITALIC TEXT - VISUAL VERIFICATION MANDATORY  
Apply <em> or <i> tags ONLY when:
  ✓ The text is VISIBLY SLANTED compared to regular upright text
  ✓ You can clearly see the text angle is different from normal text
  ✓ Another person looking at the image would agree it's italic
  ✗ Do NOT assume case citations or legal references are italic
  ✗ Do NOT make text italic because of conventions or assumptions
VISUAL TEST: Compare text angle to regular upright text. If you cannot see slanting, use plain text.
EXAMPLE: If "Powell v McFarlane" appears upright → Powell v McFarlane (plain)
EXAMPLE: If "Powell v McFarlane" is clearly slanted → <em>Powell v McFarlane</em>

RULE 3: LIST MARKERS ARE USUALLY PLAIN TEXT
DEFAULT: List markers (A., B., C., 1., 2., 3., a., b., i., ii.) are PLAIN TEXT in 99% of documents
VERIFICATION: Look at EACH marker individually and compare to body text
CRITICAL: Do NOT make markers bold unless they are OBVIOUSLY thicker than body text
EXAMPLE: If "A." has same thickness as text → <li>A. Content here</li>
EXAMPLE: If "A." is clearly thicker → <li><strong>A.</strong> Content here</li>

RULE 4: WHEN UNCERTAIN, CHOOSE PLAIN TEXT
If you have ANY doubt about whether text is bold or italic:
  → Use plain text with NO formatting tags
  → Plain text is ALWAYS better than incorrect formatting
  → Err on the side of less formatting rather than more

RULE 5: TRANSLATION ONLY CHANGES LANGUAGE
Spanish → English: Only language changes, formatting stays IDENTICAL
German → English: Only language changes, formatting stays IDENTICAL  
Any language → English: Format in source = Format in output (no additions, no removals)

🔴 MANDATORY PRE-OUTPUT VERIFICATION PROCESS:
Before providing your HTML output, perform these checks:

1. SCAN every <strong>, <b>, <em>, <i> tag in your HTML
2. For EACH tag, look back at the source image at that exact text
3. VERIFY the formatting is clearly visible in the source
4. ASK: "Would another person agree this text is bold/italic in the source?"
5. REMOVE any tag where you cannot clearly confirm the formatting
6. CHECK: Did I add ANY strikethrough (<del>, <s>)? If YES, REMOVE IT IMMEDIATELY
7. CHECK: Did I add ANY underline (<u>) where text isn't underlined? If YES, REMOVE IT
8. CHECK: Did I interpret a highlight as formatting? If YES, REMOVE THE FORMATTING

For lists - CRITICAL FORMATTING AND TRANSLATION RULES:

TRANSLATE all list items to English completely
Preserve EXACT list structure from the original image
Any list in the original image should be represented as a proper HTML list
CRITICAL: ALL CONSECUTIVE LIST ITEMS MUST BE IN THE SAME LIST STRUCTURE
PRESERVE EXACT PUNCTUATION: If original shows "a)" use "a)" NOT "a.", if shows "1)" use "1)" NOT "1."
Maintain exact indentation and hierarchy of the original list
Use <ul> for unordered lists (bullets or hyphens)
Use <ol> for ordered lists (numbers, letters, or roman numerals)
CRITICAL: List markers are usually PLAIN TEXT - do not make them bold unless visually confirmed

For tables - CRITICAL TRANSLATION RULES:

TRANSLATE all table content (headers, data, captions) to English completely
Use proper HTML table tags (<table>, <tr>, <td>, <th>)
Maintain the original table structure, including any merged cells or special formatting
Ensure all data from the original table is included without summarization
Preserve column alignment and cell borders as shown in original

For indentation and text alignment:

Use appropriate CSS styles inline (e.g., style="text-indent: 20px;")
Use text-align property for alignment (e.g., style="text-align: center;")

🔴 FINAL CRITICAL REMINDERS - 100% FORMATTING ACCURACY:

✓ ONLY translate language - formatting must match source EXACTLY
✓ Visual confirmation MANDATORY before ANY formatting tag
✓ When uncertain, ALWAYS use plain text
✓ NEVER use strikethrough - it is absolutely forbidden
✓ IGNORE all PDF highlights and annotations - they are not text formatting
✓ Bold text must be VISIBLY thicker than regular text
✓ Italic text must be VISIBLY slanted compared to regular text
✓ List markers are usually plain text - don't assume they're bold
✓ Section numbers are usually plain text - don't assume they're bold
✓ Default to plain text when in doubt
✓ Formatting accuracy is MORE important than formatting assumptions

Your output should be in HTML format only with ALL content translated to ENGLISH and formatting matching source EXACTLY. Do not include any additional tags, explanations, or extra information. The response should contain pure HTML code ready to be rendered."""
        
    def process_page_to_html(self, page_num: int, dpi: int = 450, max_retries: int = 2) -> Dict[str, Any]:  # ADDED RETRY LOGIC  
        """    
        Process a single page to HTML using vision API with retry logic.
            
        Args:    
            page_num: Page number (0-indexed)    
            dpi: Resolution for image extraction    
            max_retries: Maximum number of retry attempts for incomplete responses
            
        Returns:    
            Dict with page info and HTML content    
        """    
        print(f"  Processing page {page_num + 1}/{self.total_pages}...")
          
        for attempt in range(max_retries + 1):  
            try:    
                # Extract page as image    
                image_bytes = self.extract_page_as_image(page_num, dpi)    
                if not image_bytes:    
                    return {    
                        "page_num": page_num + 1,    
                        "status": "failed",    
                        "error": "Failed to extract image",    
                        "html_content": ""    
                    }
                
                # Compress image if needed to stay under Claude's 5MB limit
                compressed_result = self.compress_image_if_needed(image_bytes)
                if isinstance(compressed_result, tuple):
                    image_bytes, media_type = compressed_result
                else:
                    # Backward compatibility - if function returns single value
                    image_bytes = compressed_result
                    media_type = 'image/png'
                    
                # Encode image to base64    
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                # Create vision prompt    
                prompt = self.create_vision_prompt()
                    
                # Call Claude Vision API with INCREASED max_tokens  
                response = self.claude_client.messages.create(    
                    model=self.model,    
                    max_tokens=32000,  # INCREASED from 16384 to 32000 for longer pages  
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
                                    "text": prompt    
                                }    
                            ]    
                        }    
                    ]    
                )
                    
                # Extract HTML content    
                html_content = response.content[0].text.strip()
                
                # NO POST-PROCESSING - Prompt should ensure 100% accurate formatting
                # Post-processing removed to avoid introducing new issues
                
                # Check if response was truncated due to max_tokens
                if response.stop_reason == "max_tokens":
                    print(f"    ⚠ Page {page_num + 1} response truncated due to max_tokens limit!")
                    if attempt < max_retries:
                        print(f"    ⚠ Retrying with focus on complete translation... (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        print(f"    ⚠ Page {page_num + 1} still truncated after {max_retries} retries - may need higher max_tokens")
                    
                # Clean up HTML (remove markdown code blocks if present)    
                html_content = html_content.replace("```html", "").replace("```", "").strip()
                  
                # VALIDATION: Check if response seems complete  
                if len(html_content) < 100:  
                    if attempt < max_retries:  
                        print(f"    ⚠ Page {page_num + 1} response seems too short ({len(html_content)} chars), retrying... (attempt {attempt + 1}/{max_retries})")  
                        continue  
                    else:  
                        print(f"    ⚠ Page {page_num + 1} response still incomplete after {max_retries} retries")
                  
                if not html_content.strip().endswith('>'):  
                    if attempt < max_retries:  
                        print(f"    ⚠ Page {page_num + 1} HTML seems truncated (doesn't end with >), retrying... (attempt {attempt + 1}/{max_retries})")  
                        continue  
                    else:  
                        print(f"    ⚠ Page {page_num + 1} HTML still appears truncated after {max_retries} retries")
                    
                print(f"    ✓ Page {page_num + 1} processed successfully ({len(html_content)} chars)")
                    
                return {    
                    "page_num": page_num + 1,    
                    "status": "success",    
                    "html_content": html_content,    
                    "content_length": len(html_content),  
                    "attempts": attempt + 1  # Track how many attempts it took  
                }
                    
            except Exception as e:    
                if attempt < max_retries:  
                    print(f"    ⚠ Page {page_num + 1} attempt {attempt + 1} failed: {e}, retrying...")  
                    continue  
                else:  
                    print(f"    ✗ Page {page_num + 1} failed after {max_retries + 1} attempts: {e}")    
                    return {    
                        "page_num": page_num + 1,    
                        "status": "failed",    
                        "error": str(e),    
                        "html_content": f"<!-- Error processing page {page_num + 1}: {str(e)} -->",  
                        "attempts": attempt + 1  
                    }
    
    def _clean_formatting_issues(self, html_content: str) -> str:
        """
        Post-process HTML to remove incorrect formatting that Claude added.
        
        Removes bold/italic from:
        - List markers (A., B., 1., 2., (a), (i), etc.)
        - Section numbers (Article 5., Section 10.)
        - Other common false positives
        """
        import re
        
        # Patterns to clean (remove bold/strong tags)
        patterns = [
            # Uppercase letter list markers: <strong>A.</strong> or <b>A.</b>
            (r'<(?:strong|b)>([A-Z]\.)</(?:strong|b)>(?=\s)', r'\1'),
            # Lowercase letter list markers: <strong>a.</strong>
            (r'<(?:strong|b)>([a-z]\.)</(?:strong|b)>(?=\s)', r'\1'),
            # Number list markers: <strong>1.</strong>
            (r'<(?:strong|b)>(\d+\.)</(?:strong|b)>(?=\s)', r'\1'),
            # Parenthesized letters: <strong>(a)</strong>
            (r'<(?:strong|b)>\(([a-zA-Z])\)</(?:strong|b)>(?=\s)', r'(\1)'),
            # Parenthesized numbers: <strong>(1)</strong>
            (r'<(?:strong|b)>\((\d+)\)</(?:strong|b)>(?=\s)', r'(\1)'),
            # Roman numerals: <strong>(i)</strong> or <strong>(ii)</strong>
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
          
        # Should never reach here, but just in case  
        return {    
            "page_num": page_num + 1,    
            "status": "failed",    
            "error": "Max retries exceeded",    
            "html_content": f"<!-- Error processing page {page_num + 1}: Max retries exceeded -->",  
            "attempts": max_retries + 1  
        }
        
    def process_all_pages(self, start_page: int = 0, end_page: int = None, dpi: int = 450) -> List[Dict[str, Any]]:    
        """    
        Process all pages (or a range) to HTML.
            
        Args:    
            start_page: Starting page number (0-indexed)    
            end_page: Ending page number (0-indexed, None = all pages)    
            dpi: Resolution for image extraction
            
        Returns:    
            List of page results    
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
        
    def create_complete_html_document(self, page_results: List[Dict[str, Any]]) -> str:    
        """    
        Create a complete HTML document from page results.
            
        Args:    
            page_results: List of page processing results
            
        Returns:    
            Complete HTML document as string    
        """    
        timestamp = datetime.now().strftime('%B %Y')
            
        # Combine all page HTML    
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
            
        .conversion-info {{    
            background-color: #e8f4f8;    
            border: 1px solid #b8dce8;    
            border-radius: 5px;    
            padding: 15px;    
            margin-bottom: 20px;    
            font-size: 10pt;    
            color: #2c5f7a;    
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
            
        /* Table styling */    
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
            
        /* List styling - Enhanced for better punctuation preservation */    
        ul, ol {{    
            margin: 10px 0;    
            padding-left: 30px;    
        }}
            
        li {{    
            margin: 5px 0;    
        }}
          
        /* Ensure alphabetical lists preserve punctuation */  
        ol[type="a"], ol[type="A"] {{  
            list-style-type: lower-alpha;  
        }}
            
        @media print {{    
            body {{    
                background-color: white;    
            }}    
            .conversion-info {{    
                display: none;    
            }}    
            .page {{    
                box-shadow: none;    
                margin-bottom: 0;    
            }}    
        }}    
    </style>    
</head>    
<body>    
    
        
    <div class="document-content">    
{combined_content}    
    </div>    
</body>    
</html>'''
        
    def convert_pdf_to_html(    
        self,     
        pdf_path: str = None,     
        pdf_bytes: bytes = None,    
        output_file: str = None,    
        start_page: int = 0,    
        end_page: int = None,    
        dpi: int = 450  # INCREASED DEFAULT DPI  
    ) -> Dict[str, Any]:    
        """    
        Complete PDF to HTML conversion workflow.
            
        Args:    
            pdf_path: Path to PDF file    
            pdf_bytes: PDF content as bytes    
            output_file: Output HTML file path    
            start_page: Starting page (0-indexed)    
            end_page: Ending page (0-indexed, None = all)    
            dpi: Image resolution (increased default to 450)
            
        Returns:    
            Dict with conversion results    
        """    
        try:    
            # Load PDF    
            if not self.load_pdf(pdf_path, pdf_bytes):    
                return {"success": False, "error": "Failed to load PDF"}
                
            # Process pages    
            print(f"\n📄 Processing pages {start_page + 1} to {end_page + 1 if end_page else self.total_pages}...")    
            page_results = self.process_all_pages(start_page, end_page, dpi)
                
            # Create complete HTML    
            html_content = self.create_complete_html_document(page_results)
                
            # Generate output filename if not provided    
            if not output_file:    
                if pdf_path:    
                    base_name = os.path.splitext(os.path.basename(pdf_path))[0]    
                else:    
                    base_name = "document"    
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    
                output_file = f"{base_name}_vision_{timestamp}.html"
                
            # Save HTML    
            with open(output_file, 'w', encoding='utf-8') as f:    
                f.write(html_content)
                
            # Calculate statistics    
            successful = sum(1 for r in page_results if r['status'] == 'success')    
            failed = len(page_results) - successful
                
            print(f"\n✅ Conversion complete!")    
            print(f"  Output: {output_file}")    
            print(f"  Pages processed: {len(page_results)}")    
            print(f"  Successful: {successful}")    
            print(f"  Failed: {failed}")
                
            return {    
                "success": True,    
                "output_file": output_file,    
                "total_pages": len(page_results),    
                "successful_pages": successful,    
                "failed_pages": failed,    
                "page_results": page_results    
            }
                
        except Exception as e:    
            print(f"\n✗ Conversion failed: {e}")    
            return {"success": False, "error": str(e)}
            
        finally:    
            if self.pdf_document:    
                self.pdf_document.close()

    
# ============================================================================    
# MAIN EXECUTION    
# ============================================================================
    
def main():    
    """Main execution function"""    
    import sys
        
    # Parse arguments    
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--f=')]
        
    if len(args) < 1:    
        print("Usage: python script.py <pdf_path> [output_file] [--start=0] [--end=10] [--dpi=450]")
        return
        
    pdf_path = args[0]    
    output_file = args[1] if len(args) > 1 else None
        
    # Parse optional arguments    
    start_page = 0    
    end_page = None    
    dpi = 450
        
    for arg in sys.argv[1:]:    
        if arg.startswith('--start='):    
            start_page = int(arg.split('=')[1])    
        elif arg.startswith('--end='):    
            end_page = int(arg.split('=')[1])    
        elif arg.startswith('--dpi='):    
            dpi = int(arg.split('=')[1])
        
    # Initialize converter
    print("="*80)
    print("📄 PDF TO HTML CONVERTER (Vision-Based OCR)")
    print("="*80)
    
    converter = ImageBasedPDFConverter()
    result = converter.convert_pdf_to_html(    
        pdf_path=pdf_path,    
        output_file=output_file,    
        start_page=start_page,    
        end_page=end_page,    
        dpi=dpi    
    )
        
    if result['success']:    
        print(f"\n🎉 Success! Check output: {result['output_file']}")    
    else:    
        print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")

    
if __name__ == "__main__":    
    main()
