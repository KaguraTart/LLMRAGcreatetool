"""
Markdown 解析器
支持：纯 Markdown 文件的语义分块
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MarkdownExtractionResult:
    """Markdown 解析结果"""
    file_path: str
    title: str = ""
    sections: list[dict] = field(default_factory=list)  # [{title, level, content}]
    full_text: str = ""
    hierarchy: dict = field(default_factory=dict)


class MarkdownExtractor:
    """
    Markdown 解析器
    
    策略：
    1. 识别标题层级（# ## ###）
    2. 按标题切分章节
    3. 识别代码块、表格、列表等特殊元素
    """
    
    # 标题正则
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    # 代码块正则
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```|`[^`]+`', re.MULTILINE)
    # 表格正则
    TABLE_PATTERN = re.compile(r'\|.+\|\n\|[-| :]+\|\n((?:\|.+\|\n)*)', re.MULTILINE)
    
    def extract(self, file_path: str) -> MarkdownExtractionResult:
        """提取 Markdown 文件"""
        file_path = Path(file_path)
        result = MarkdownExtractionResult(file_path=str(file_path))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"读取 Markdown 失败: {e}")
            return result
        
        result.full_text = content
        
        # 提取标题
        headings = list(self.HEADING_PATTERN.finditer(content))
        if headings:
            first_heading = headings[0].group(2).strip()
            result.title = first_heading
        
        # 按标题切分章节
        sections = []
        
        for i, match in enumerate(headings):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.end()
            
            # 找到下一个标题的位置
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
        
        # 构建层级树
        result.hierarchy = self._build_hierarchy_tree(sections)
        
        logger.info(f"Markdown 解析完成: {len(sections)} 个章节")
        
        return result
    
    def _build_hierarchy_tree(self, sections: list[dict]) -> dict:
        """将扁平章节列表构建为层级树"""
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
            
            # 找到合适的父节点
            while stack and stack[-1]['level'] >= section['level']:
                stack.pop()
            
            stack[-1]['children'].append(node)
            stack.append(node)
        
        return root
