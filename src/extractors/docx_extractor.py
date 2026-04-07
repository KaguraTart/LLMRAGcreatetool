"""
Word Document Parser
Supports: .docx format (.doc requires conversion)
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DOCXExtractionResult:
    """Word document parsing result"""
    file_path: str
    title: str = ""
    paragraphs: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)  # Markdown format
    full_text: str = ""
    metadata: dict = field(default_factory=dict)


class DOCXExtractor:
    """
    Word Document Parser
    
    Advantages: Native support for heading hierarchy, tables, structured content
    Disadvantages: Does not support .doc (requires LibreOffice conversion)
    """
    
    def extract(self, file_path: str) -> DOCXExtractionResult:
        """Extract Word document content"""
        file_path = Path(file_path)
        result = DOCXExtractionResult(file_path=str(file_path))
        
        try:
            from docx import Document
        except ImportError:
            logger.error("python-docx not installed: pip install python-docx")
            return result
        
        try:
            doc = Document(file_path)
            
            # Extract title
            if doc.core_properties.title:
                result.title = doc.core_properties.title
            
            # Extract paragraphs
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    result.paragraphs.append(text)
            
            # Extract tables
            for table in doc.tables:
                table_md = self._table_to_markdown(table)
                if table_md:
                    result.tables.append(table_md)
            
            # Merge full text
            result.full_text = "\n\n".join(result.paragraphs)
            
            # Metadata
            core = doc.core_properties
            result.metadata = {
                'author': core.author or "",
                'created': str(core.created) if core.created else "",
                'modified': str(core.modified) if core.modified else "",
                'subject': core.subject or "",
            }
            
            logger.info(f"DOCX parsing complete: {len(result.paragraphs)} paragraphs, "
                       f"{len(result.tables)} tables")
            
        except Exception as e:
            logger.error(f"DOCX parsing failed: {e}")
        
        return result
    
    def _table_to_markdown(self, table) -> str:
        """Convert Word table to Markdown"""
        rows = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip() for cell in row.cells]
            row_str = "| " + " | ".join(cells) + " |"
            rows.append(row_str)
            
            if i == 0:
                rows.append("|" + "|".join(["---"] * len(cells)) + "|")
        
        return "\n".join(rows) if rows else ""
