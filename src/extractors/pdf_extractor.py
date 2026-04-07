"""
PDF Parser
Supports: text extraction / table extraction / image extraction / OCR / hierarchical structure recognition
"""

import io
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import pymupdf  # fitz
import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class PDFPage:
    """Single page PDF parsing result"""
    page_number: int
    text: str
    blocks: list[dict] = field(default_factory=list)
    tables: list[list[list]] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    is_scanned: bool = False
    source_method: str = "pymupdf"  # pymupdf / pdfplumber / ocr


@dataclass
class PDFExtractionResult:
    """PDF document parsing result"""
    file_path: str
    title: str = ""
    total_pages: int = 0
    pages: list[PDFPage] = field(default_factory=list)
    full_text: str = ""
    hierarchy: dict = field(default_factory=dict)  # Heading hierarchy structure


class PDFExtractor:
    """
    Multi-layer PDF Parser
    
    Parsing strategy:
    Layer 1: PyMuPDF (fast text extraction, preserves layout information)
    Layer 2: pdfplumber (table extraction, preserves character-level coordinates)
    Layer 3: OCR (scanned documents / low-quality text pages)
    """
    
    def __init__(
        self,
        extract_images: bool = True,
        extract_tables: bool = True,
        ocr_enabled: bool = True,
        min_text_length: int = 50,
    ):
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.ocr_enabled = ocr_enabled
        self.min_text_length = min_text_length
        self._ocr_processor = None  # Lazy load
    
    def extract(self, file_path: str) -> PDFExtractionResult:
        """
        Extract complete content from PDF
        
        Args:
            file_path: PDF file path
            
        Returns:
            PDFExtractionResult: Parsing result
        """
        file_path = Path(file_path)
        logger.info(f"Starting PDF parsing: {file_path.name}")
        
        result = PDFExtractionResult(file_path=str(file_path))
        
        # Get total page count
        with pymupdf.open(file_path) as doc:
            result.total_pages = len(doc)
        
        # Try to extract title (from metadata or filename)
        result.title = self._extract_title(file_path)
        
        # Multi-threaded parallel page parsing
        pages = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._extract_page, file_path, i): i
                for i in range(result.total_pages)
            }
            
            for i in range(result.total_pages):
                page = futures[i].result()
                pages.append(page)
        
        pages.sort(key=lambda p: p.page_number)
        result.pages = pages
        
        # Merge full text
        result.full_text = "\n\n".join([
            f"[Page {p.page_number}]\n{p.text}"
            for p in pages if p.text.strip()
        ])
        
        # Rebuild hierarchy structure
        result.hierarchy = self._rebuild_hierarchy(pages)
        
        logger.info(f"PDF parsing complete: {result.total_pages} pages, "
                   f"extracted {len(result.full_text)} characters of text")
        
        return result
    
    def _extract_page(self, file_path: Path, page_num: int) -> PDFPage:
        """Parse single PDF page"""
        page = PDFPage(page_number=page_num + 1)
        
        # --- PyMuPDF text extraction ---
        try:
            with pymupdf.open(file_path) as doc:
                muvm_page = doc[page_num]
                
                # Get text (preserve block structure)
                blocks = muvm_page.get_text("dict")["blocks"]
                page.blocks = blocks
                
                text = muvm_page.get_text("text")
                page.text = text
                page.source_method = "pymupdf"
                
                # Get image information
                if self.extract_images:
                    image_list = muvm_page.get_images(full=True)
                    for img in image_list:
                        xref = img[0]
                        pix = muvm_page.parent_xref(xref)
                        
                        try:
                            pixmap = pymupdf.Pixmap(muvm_page.parent, xref)
                            if pixmap.n > 4:
                                pixmap = pymupdf.Pixmap(
                                    pymupdf.csRGB, pixmap
                                )
                            img_bytes = pixmap.tobytes("png")
                            page.images.append({
                                'xref': xref,
                                'page': page_num,
                                'bytes': img_bytes,
                                'width': pixmap.width,
                                'height': pixmap.height,
                            })
                        except Exception:
                            continue
                
                # Determine if it's a scanned page (no text or too little text)
                if len(text.strip()) < self.min_text_length:
                    page.is_scanned = True
                
        except Exception as e:
            logger.warning(f"PyMuPDF parsing page {page_num} failed: {e}")
            text = ""
            page.is_scanned = True
        
        # --- pdfplumber table extraction ---
        if self.extract_tables:
            try:
                with pdfplumber.open(file_path) as pdf:
                    if page_num < len(pdf.pages):
                        plumb_page = pdf.pages[page_num]
                        
                        # Extract tables
                        tables = plumb_page.extract_tables()
                        if tables:
                            page.tables = [t for t in tables if t]
                        
                        # If pdfplumber text is longer than PyMuPDF, use it
                        plumb_text = plumb_page.extract_text() or ""
                        if len(plumb_text) > len(text):
                            page.text = plumb_text
                            page.source_method = "pdfplumber"
            except Exception as e:
                logger.debug(f"pdfplumber table extraction failed: {e}")
        
        # --- OCR (scanned pages) ---
        if self.ocr_enabled and page.is_scanned:
            ocr_text = self._ocr_page(file_path, page_num)
            if ocr_text:
                page.text = ocr_text
                page.is_scanned = False
                page.source_method = "ocr"
        
        return page
    
    def _ocr_page(self, file_path: Path, page_num: int) -> Optional[str]:
        """Perform OCR on scanned page"""
        if self._ocr_processor is None:
            try:
                from .ocr_extractor import OCRProcessor
                self._ocr_processor = OCRProcessor()
            except ImportError:
                logger.warning("OCR processor not installed, skipping OCR")
                return None
        
        try:
            return self._ocr_processor.ocr_pdf_page(file_path, page_num)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return None
    
    def _extract_title(self, file_path: Path) -> str:
        """Extract PDF metadata title"""
        try:
            with pymupdf.open(file_path) as doc:
                meta = doc.metadata
                title = meta.get("title", "").strip()
                if not title:
                    # Try to extract from filename
                    title = file_path.stem
                return title
        except:
            return file_path.stem
    
    def _rebuild_hierarchy(self, pages: list[PDFPage]) -> dict:
        """Identify heading hierarchy based on font size"""
        headings = []
        
        for page in pages:
            if not page.blocks:
                continue
            
            for block in page.blocks:
                if block.get("type") != 0:  # Skip non-text blocks
                    continue
                
                lines = block.get("lines", [])
                if not lines:
                    continue
                
                # Use the largest font in the block's first line as basis for heading judgment
                max_size = 0
                block_text = ""
                for line in lines:
                    for span in line.get("spans", []):
                        size = span.get("size", 12)
                        if size > max_size:
                            max_size = size
                            block_text = span.get("text", "").strip()
                
                # Font > 14pt is considered a heading
                if max_size > 14 and len(block_text) > 2 and len(block_text) < 100:
                    level = 1 if max_size > 18 else 2
                    headings.append({
                        'text': block_text,
                        'level': level,
                        'page': page.page_number,
                        'font_size': max_size,
                    })
        
        return {'headings': headings}
    
    def get_page_image(self, file_path: str, page_num: int) -> Optional[bytes]:
        """Get image of specified page (for multimodal understanding)"""
        try:
            with pymupdf.open(file_path) as doc:
                page = doc[page_num]
                mat = pymupdf.Matrix(2, 2)  # 2x zoom
                pix = page.get_pixmap(matrix=mat)
                return pix.tobytes("png")
        except Exception as e:
            logger.warning(f"Failed to get page image: {e}")
            return None
    
    def extract_table_as_markdown(
        self, 
        table: list[list]
    ) -> str:
        """Convert table data to Markdown format"""
        if not table or not table[0]:
            return ""
        
        lines = []
        header = table[0]
        lines.append("| " + " | ".join(str(c).strip() if c else "" for c in header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        
        for row in table[1:]:
            lines.append(
                "| " + " | ".join(str(c).strip() if c else "" for c in row) + " |"
            )
        
        return "\n".join(lines)
