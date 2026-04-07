"""
OCR Processor
Supports: Tesseract + EasyOCR dual engine
Used for: scanned PDFs, image text recognition
"""

import logging
from pathlib import Path
from typing import Optional
from io import BytesIO

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    OCR Processor
    
    Dual engine strategy:
    - EasyOCR: Good multilingual support, GPU acceleration
    - Tesseract: Easy to install, Chinese requires additional language pack
    """
    
    def __init__(
        self,
        engine: str = "easyocr",  # easyocr / tesseract / auto
        languages: list[str] = None,
        gpu: bool = True,
    ):
        self.engine = engine
        self.languages = languages or ['ch_sim', 'en']  # Simplified Chinese + English
        self.gpu = gpu
        self._reader = None
    
    def _get_easyocr_reader(self):
        """Lazy load EasyOCR engine"""
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu,
                    verbose=False
                )
                logger.info("EasyOCR engine initialized")
            except ImportError:
                raise ImportError(
                    "EasyOCR not installed: pip install easyocr"
                )
        return self._reader
    
    def _get_tesseract(self):
        """Lazy load Tesseract"""
        if self._reader is None:
            try:
                import pytesseract
                from PIL import Image
                self._reader = {'pytesseract': pytesseract, 'PIL': Image}
                logger.info("Tesseract OCR engine initialized")
            except ImportError:
                raise ImportError(
                    "pytesseract not installed: pip install pytesseract"
                )
        return self._reader
    
    def ocr_image(
        self, 
        image_bytes: bytes,
        return_confidence: bool = False
    ):
        """
        Perform OCR on image
        
        Args:
            image_bytes: Image byte data
            return_confidence: Whether to return confidence score
            
        Returns:
            Recognized text, or (text, confidence) tuple
        """
        if self.engine in ("easyocr", "auto"):
            return self._ocr_easyocr(image_bytes, return_confidence)
        else:
            return self._ocr_tesseract(image_bytes, return_confidence)
    
    def _ocr_easyocr(
        self, 
        image_bytes: bytes,
        return_confidence: bool
    ):
        """EasyOCR engine"""
        try:
            reader = self._get_easyocr_reader()
        except ImportError:
            logger.warning("EasyOCR not available, trying Tesseract")
            return self._ocr_tesseract(image_bytes, return_confidence)
        
        from PIL import Image
        import numpy as np
        
        img = Image.open(BytesIO(image_bytes))
        img_array = np.array(img)
        
        results = reader.readtext(img_array)
        
        if return_confidence:
            lines = []
            total_conf = 0
            count = 0
            for (bbox, text, conf) in results:
                if text.strip():
                    lines.append(text)
                    total_conf += conf
                    count += 1
            text = "\n".join(lines)
            avg_conf = total_conf / count if count > 0 else 0
            return text, avg_conf
        else:
            lines = []
            for (bbox, text, conf) in results:
                if text.strip():
                    lines.append(text)
            return "\n".join(lines)
    
    def _ocr_tesseract(
        self, 
        image_bytes: bytes,
        return_confidence: bool
    ):
        """Tesseract engine"""
        try:
            engine = self._get_tesseract()
        except ImportError:
            raise RuntimeError("All OCR engines are unavailable")
        
        pytesseract = engine['pytesseract']
        Image = engine['PIL']
        
        img = Image.open(BytesIO(image_bytes))
        
        if return_confidence:
            data = pytesseract.image_to_data(img, output_type=pytesseract.OUTPUT)
            lines = []
            confidences = []
            for d in data.splitlines():
                parts = d.split('\t')
                if len(parts) >= 12:
                    text = parts[11].strip()
                    conf = float(parts[10]) if parts[10] else 0
                    if text:
                        lines.append(text)
                        confidences.append(conf)
            text = "\n".join(lines)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            return text, avg_conf
        else:
            return pytesseract.image_to_string(img, lang='chi_sim+eng')
    
    def ocr_pdf_page(self, pdf_path: str, page_num: int) -> Optional[str]:
        """
        Perform OCR on specified PDF page
        
        Args:
            pdf_path: PDF file path
            page_num: Page number (starting from 0)
            
        Returns:
            Recognized text, None if failed
        """
        try:
            import pymupdf
            
            with pymupdf.open(pdf_path) as doc:
                if page_num >= len(doc):
                    return None
                
                page = doc[page_num]
                
                # Render page as image
                mat = pymupdf.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                image_bytes = pix.tobytes("png")
            
            text = self.ocr_image(image_bytes)
            
            if text and len(text.strip()) > 10:
                logger.debug(f"OCR success: PDF page {page_num}, "
                           f"recognized {len(text)} characters")
                return text
            
            return None
            
        except Exception as e:
            logger.warning(f"PDF OCR failed (page {page_num}): {e}")
            return None
    
    def batch_ocr_images(
        self, 
        image_paths: list[str],
        max_workers: int = 4
    ) -> list[Optional[str]]:
        """
        Batch OCR multiple images (concurrent)
        
        Args:
            image_paths: List of image paths
            max_workers: Maximum concurrency
            
        Returns:
            List of recognition results (None for failed items)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = [None] * len(image_paths)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.ocr_image, self._read_image(path)): path
                for path in image_paths
            }
            
            for future in as_completed(futures):
                idx = image_paths.index(futures[future])
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Batch OCR failed: {e}")
        
        return results
    
    def _read_image(self, path: str) -> bytes:
        """Read image file"""
        with open(path, 'rb') as f:
            return f.read()
