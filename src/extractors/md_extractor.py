"""
Markdown Parser
Supports: Semantic chunking of plain Markdown files
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MarkdownExtractionResult:
    """Markdown parsing result"""
    file_path: str
    title: str = ""
    sections: list[dict] = field(default_factory=list)  # [{title, level, content}]
    full_text: str = ""
    hierarchy: dict = field(default_factory=dict)


class MarkdownExtractor:
    """
    Markdown Parser
    
    Strategy:
    1. Identify heading hierarchy (# ## ###)
    2. Split sections by headings
    3. Identify special elements like code blocks, tables, lists, etc.
    """
    
    # Heading regex
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    # Code block regex
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```|`[^`]+`', re.MULTILINE)
    # Table regex
    TABLE_PATTERN = re.compile(r'\|.+\|\n\|[-| :]+\|\n((?:\|.+\|\n)*)', re.MULTILINE)
    
    def extract(self, file_path: str) -> MarkdownExtractionResult:
        """Extract Markdown file"""
        file_path = Path(file_path)
        result = MarkdownExtractionResult(file_path=str(file_path))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read Markdown: {e}")
            return result
        
        result.full_text = content
        
        # Extract headings
        headings = list(self.HEADING_PATTERN.finditer(content))
        if headings:
            first_heading = headings[0].group(2).strip()
            result.title = first_heading
        
        # Split sections by headings
        sections = []
        
        for i, match in enumerate(headings):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.end()
            
            # Find position of next heading
            end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
            
            section_text = content[start:end].strip()
            
            sections.append({
                'title': title,
                'level': level,
                'content': section_text,
                'start': start,
                'end': end,
            })
        
        result.sections = sections
        
        # Build hierarchy tree
        result.hierarchy = self._build_hierarchy_tree(sections)
        
        logger.info(f"Markdown parsing complete: {len(sections)} sections")
        
        return result
    
    def _build_hierarchy_tree(self, sections: list[dict]) -> dict:
        """Build flat section list into hierarchy tree"""
        if not sections:
            return {}
        
        root = {'title': 'root', 'level': 0, 'children': []}
        stack = [root]
        
        for section in sections:
            node = {
                'title': section['title'],
                'level': section['level'],
                'content_preview': section['content'][:200],
                'children': []
            }
            
            # Find appropriate parent node
            while stack and stack[-1]['level'] >= section['level']:
                stack.pop()
            
            stack[-1]['children'].append(node)
            stack.append(node)
        
        return root
