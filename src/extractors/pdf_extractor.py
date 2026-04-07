"""
PDF 解析器
支持：文字提取 / 表格提取 / 图片提取 / OCR / 层级结构识别
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
    """PDF 单页解析结果"""
    page_number: int
    text: str
    blocks: list[dict] = field(default_factory=list)
    tables: list[list[list]] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    is_scanned: bool = False
    source_method: str = "pymupdf"  # pymupdf / pdfplumber / ocr


@dataclass
class PDFExtractionResult:
    """PDF 文档解析结果"""
    file_path: str
    title: str = ""
    total_pages: int = 0
    pages: list[PDFPage] = field(default_factory=list)
    full_text: str = ""
    hierarchy: dict = field(default_factory=dict)  # 标题层级结构


class PDFExtractor:
    """
    多层 PDF 解析器
    
    解析策略：
    Layer 1: PyMuPDF（快速文字提取，保留布局信息）
    Layer 2: pdfplumber（表格提取，保留字符级坐标）
    Layer 3: OCR（扫描件 / 低质量文字页面）
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
        self._ocr_processor = None  # 延迟加载
    
    def extract(self, file_path: str) -> PDFExtractionResult:
        """
        提取 PDF 的完整内容
        
        Args:
            file_path: PDF 文件路径
            
        Returns:
            PDFExtractionResult: 解析结果
        """
        file_path = Path(file_path)
        logger.info(f"开始解析 PDF: {file_path.name}")
        
        result = PDFExtractionResult(file_path=str(file_path))
        
        # 获取总页数
        with pymupdf.open(file_path) as doc:
            result.total_pages = len(doc)
        
        # 尝试提取标题（从元数据或文件名）
        result.title = self._extract_title(file_path)
        
        # 多线程并行解析每页
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
        
        # 合并全文
        result.full_text = "\n\n".join([
            f"[Page {p.page_number}]\n{p.text}"
            for p in pages if p.text.strip()
        ])
        
        # 重建层级结构
        result.hierarchy = self._rebuild_hierarchy(pages)
        
        logger.info(f"PDF 解析完成: {result.total_pages} 页, "
                   f"提取文字 {len(result.full_text)} 字符")
        
        return result
    
    def _extract_page(self, file_path: Path, page_num: int) -> PDFPage:
        """解析单页 PDF"""
        page = PDFPage(page_number=page_num + 1)
        
        # --- PyMuPDF 文字提取 ---
        try:
            with pymupdf.open(file_path) as doc:
                muvm_page = doc[page_num]
                
                # 获取文字（保留块结构）
                blocks = muvm_page.get_text("dict")["blocks"]
                page.blocks = blocks
                
                text = muvm_page.get_text("text")
                page.text = text
                page.source_method = "pymupdf"
                
                # 获取图片信息
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
                
                # 判断是否扫描页（无文字或文字过少）
                if len(text.strip()) < self.min_text_length:
                    page.is_scanned = True
                
        except Exception as e:
            logger.warning(f"PyMuPDF 解析第 {page_num} 页失败: {e}")
            text = ""
            page.is_scanned = True
        
        # --- pdfplumber 表格提取 ---
        if self.extract_tables:
            try:
                with pdfplumber.open(file_path) as pdf:
                    if page_num < len(pdf.pages):
                        plumb_page = pdf.pages[page_num]
                        
                        # 提取表格
                        tables = plumb_page.extract_tables()
                        if tables:
                            page.tables = [t for t in tables if t]
                        
                        # 如果 pdfplumber 的文字比 PyMuPDF 更长，使用它
                        plumb_text = plumb_page.extract_text() or ""
                        if len(plumb_text) > len(text):
                            page.text = plumb_text
                            page.source_method = "pdfplumber"
            except Exception as e:
                logger.debug(f"pdfplumber 表格提取失败: {e}")
        
        # --- OCR（扫描件）---
        if self.ocr_enabled and page.is_scanned:
            ocr_text = self._ocr_page(file_path, page_num)
            if ocr_text:
                page.text = ocr_text
                page.is_scanned = False
                page.source_method = "ocr"
        
        return page
    
    def _ocr_page(self, file_path: Path, page_num: int) -> Optional[str]:
        """对扫描页执行 OCR"""
        if self._ocr_processor is None:
            try:
                from .ocr_extractor import OCRProcessor
                self._ocr_processor = OCRProcessor()
            except ImportError:
                logger.warning("OCR 处理器未安装，跳过 OCR")
                return None
        
        try:
            return self._ocr_processor.ocr_pdf_page(file_path, page_num)
        except Exception as e:
            logger.warning(f"OCR 失败: {e}")
            return None
    
    def _extract_title(self, file_path: Path) -> str:
        """提取 PDF 元数据标题"""
        try:
            with pymupdf.open(file_path) as doc:
                meta = doc.metadata
                title = meta.get("title", "").strip()
                if not title:
                    # 尝试从文件名提取
                    title = file_path.stem
                return title
        except:
            return file_path.stem
    
    def _rebuild_hierarchy(self, pages: list[PDFPage]) -> dict:
        """基于字体大小识别标题层级"""
        headings = []
        
        for page in pages:
            if not page.blocks:
                continue
            
            for block in page.blocks:
                if block.get("type") != 0:  # 非文字块跳过
                    continue
                
                lines = block.get("lines", [])
                if not lines:
                    continue
                
                # 取块中第一行的大字体作为标题判断依据
                max_size = 0
                block_text = ""
                for line in lines:
                    for span in line.get("spans", []):
                        size = span.get("size", 12)
                        if size > max_size:
                            max_size = size
                            block_text = span.get("text", "").strip()
                
                # 字体 > 14pt 视为标题
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
        """获取指定页面的图像（用于多模态理解）"""
        try:
            with pymupdf.open(file_path) as doc:
                page = doc[page_num]
                mat = pymupdf.Matrix(2, 2)  # 2x 缩放
                pix = page.get_pixmap(matrix=mat)
                return pix.tobytes("png")
        except Exception as e:
            logger.warning(f"获取页面图像失败: {e}")
            return None
    
    def extract_table_as_markdown(
        self, 
        table: list[list]
    ) -> str:
        """将表格数据转换为 Markdown 格式"""
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
