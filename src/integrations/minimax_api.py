"""
MiniMax API 封装
支持：文本生成 / 函数调用（实体抽取）/ 图像理解
"""

import os
import json
import logging
import base64
from io import BytesIO
from typing import Optional, Any

logger = logging.getLogger(__name__)


class MiniMaxClient:
    """
    MiniMax API 客户端
    
    功能：
    1. 文本生成（Entity / Relation 抽取、分类、质量评分）
    2. 图像理解（图表描述、多模态理解）
    3. 函数调用（Tool Use）
    """
    
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.minimaxi.com",
        model: str = "MiniMax-Text-01",
        vision_model: str = "MiniMax-Hailuo-VL-01",
        embedding_model: str = "embo-01",
        timeout: int = 60,
    ):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.vision_model = vision_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        
        if not self.api_key:
            logger.warning("MINIMAX_API_KEY 未设置，部分功能不可用")
    
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """
        文本生成
        
        Args:
            prompt: 用户提示
            system: 系统提示
            temperature: 采样温度
            max_tokens: 最大生成长度
            json_mode: 是否返回 JSON
            
        Returns:
            生成的文本
        """
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY 未设置")
        
        import requests
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            
            return data["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"MiniMax API 调用失败: {e}")
            raise
    
    def generate_with_functions(
        self,
        prompt: str,
        functions: list[dict],
        system: str = "",
        temperature: float = 0.3,
    ) -> dict:
        """
        函数调用（Tool Use）
        
        用于：结构化实体抽取、关系抽取
        
        Args:
            prompt: 用户提示
            functions: 函数定义列表（OpenAI format）
            system: 系统提示
            
        Returns:
            函数调用结果（parsed）
        """
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY 未设置")
        
        import requests
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": [{"type": "function", "function": f} for f in functions],
            "tool_choice": "auto",
            "temperature": temperature,
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            
            msg = data["choices"][0]["message"]
            
            if "tool_calls" in msg:
                tool_call = msg["tool_calls"][0]
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                return {"function": func_name, "arguments": func_args}
            else:
                return {"content": msg.get("content", "")}
        
        except Exception as e:
            logger.error(f"MiniMax 函数调用失败: {e}")
            raise
    
    def understand_image(
        self,
        image_bytes: bytes,
        prompt: str = "请详细描述这张图片的内容",
        detail: str = "high",
    ) -> str:
        """
        图像理解（多模态）
        
        用于：图表描述、扫描件内容提取、复杂页面理解
        
        Args:
            image_bytes: 图片字节数据
            prompt: 询问提示
            detail: low / high（高分辨率=更多细节）
            
        Returns:
            图片描述文本
        """
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY 未设置")
        
        import requests
        
        # Base64 编码
        b64_image = base64.b64encode(image_bytes).decode()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                            "detail": detail,
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.vision_model,
            "messages": messages,
            "max_tokens": 4096,
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            
            return data["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"MiniMax 图像理解失败: {e}")
            raise
    
    def extract_entities(
        self,
        text: str,
        schema: dict,
    ) -> dict:
        """
        实体+关系抽取（通过函数调用）
        
        Args:
            text: 待抽取文本
            schema: 抽取 Schema
            
        Returns:
            {"entities": [...], "relations": [...]}
        """
        entity_type = schema.get("entities", [])
        relation_type = schema.get("relations", [])
        
        entity_def = "\n".join([f"- {e}: 实体类型" for e in entity_type])
        relation_def = "\n".join([f"- {r}: 关系类型" for r in relation_type])
        
        functions = [
            {
                "name": "extract_knowledge",
                "description": "从文本中抽取实体和关系",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "description": "抽取的实体列表",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                    "description": {"type": "string"}
                                }
                            }
                        },
                        "relations": {
                            "type": "array",
                            "description": "抽取的关系列表",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "target": {"type": "string"},
                                    "relation": {"type": "string"}
                                }
                            }
                        }
                    },
                    "required": ["entities", "relations"]
                }
            }
        ]
        
        prompt = f"""请从以下文本中抽取实体和关系。

实体类型：
{entity_def}

关系类型：
{relation_def}

文本：
---
{text[:4000]}
---

请调用 extract_knowledge 函数输出结果。"""
        
        system = "你是一个专业的知识抽取系统，只输出 JSON，不输出其他内容。"
        
        result = self.generate_with_functions(prompt, functions, system=system)
        
        if "function" in result and result["function"] == "extract_knowledge":
            return result["arguments"]
        else:
            return {"entities": [], "relations": []}
    
    def classify(
        self,
        text: str,
        categories: list[str],
    ) -> dict:
        """
        文本分类（LLM）
        
        Returns:
            {"category": "分类结果", "confidence": 0.0-1.0}
        """
        prompt = f"""请将以下文本分类到最合适的类别。

可选类别：{', '.join(categories)}

文本：
---
{text[:2000]}
---

输出 JSON：
{{"category": "类别名", "confidence": 0.0-1.0}}"""
        
        system = "你是文本分类专家，只输出 JSON。"
        
        response = self.generate(prompt, system=system, temperature=0.3)
        
        try:
            import re
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                return json.loads(match.group())
        except:
            pass
        
        return {"category": categories[0] if categories else "unknown", "confidence": 0.0}
