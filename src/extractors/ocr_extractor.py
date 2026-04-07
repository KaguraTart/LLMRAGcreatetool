"""
OCR 处理器
支持：Tesseract + EasyOCR 双引擎
用于：扫描件 PDF、图片文字识别
"""

import logging
from pathlib import Path
from typing import Optional
from io import BytesIO

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    OCR 处理器
    
    双引擎策略：
    - EasyOCR: 多语言支持好，GPU 加速
    - Tesseract: 安装简单，中文需额外语言包
    """
    
    def __init__(
        self,
        engine: str = "easyocr",  # easyocr / tesseract / auto
        languages: list[str] = None,
        gpu: bool = True,
    ):
        self.engine = engine
        self.languages = languages or ['ch_sim', 'en']  # 中文简体 + 英文
        self.gpu = gpu
        self._reader = None
    
    def _get_easyocr_reader(self):
        """延迟加载 EasyOCR 引擎"""
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu,
                    verbose=False
                )
                logger.info("EasyOCR 引擎初始化完成")
            except ImportError:
                raise ImportError(
                    "EasyOCR 未安装: pip install easyocr"
                )
        return self._reader
    
    def _get_tesseract(self):
        """延迟加载 Tesseract"""
        if self._reader is None:
            try:
                import pytesseract
                from PIL import Image
                self._reader = {'pytesseract': pytesseract, 'PIL': Image}
                logger.info("Tesseract OCR 引擎初始化完成")
            except ImportError:
                raise ImportError(
                    "pytesseract 未安装: pip install pytesseract"
                )
        return self._reader
    
    def ocr_image(
        self, 
        image_bytes: bytes,
        return_confidence: bool = False
    ):
        """
        对图片执行 OCR
        
        Args:
            image_bytes: 图片字节数据
            return_confidence: 是否返回置信度
            
        Returns:
            识别的文字，或 (文字, 置信度) 元组
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
        """EasyOCR 引擎"""
        try:
            reader = self._get_easyocr_reader()
        except ImportError:
            logger.warning("EasyOCR 不可用，尝试 Tesseract")
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
        """Tesseract 引擎"""
        try:
            engine = self._get_tesseract()
        except ImportError:
            raise RuntimeError("所有 OCR 引擎均不可用")
        
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
        对 PDF 指定页面执行 OCR
        
        Args:
            pdf_path: PDF 文件路径
            page_num: 页码（从 0 开始）
            
        Returns:
            识别的文字，失败返回 None
        """
        try:
            import pymupdf
            
            with pymupdf.open(pdf_path) as doc:
                if page_num >= len(doc):
                    return None
                
                page = doc[page_num]
                
                # 将页面渲染为图像
                mat = pymupdf.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                image_bytes = pix.tobytes("png")
            
            text = self.ocr_image(image_bytes)
            
            if text and len(text.strip()) > 10:
                logger.debug(f"OCR 成功: PDF page {page_num}, "
                           f"识别 {len(text)} 字符")
                return text
            
            return None
            
        except Exception as e:
            logger.warning(f"PDF OCR 失败 (page {page_num}): {e}")
            return None
    
    def batch_ocr_images(
        self, 
        image_paths: list[str],
        max_workers: int = 4
    ) -> list[Optional[str]]:
        """
        批量 OCR 多张图片（并发）
        
        Args:
            image_paths: 图片路径列表
            max_workers: 最大并发数
            
        Returns:
            识别结果列表（失败项为 None）
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
                    logger.warning(f"批量 OCR 失败: {e}")
        
        return results
    
    def _read_image(self, path: str) -> bytes:
        """读取图片文件"""
        with open(path, 'rb') as f:
            return f.read()
