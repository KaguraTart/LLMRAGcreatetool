"""
MiniMax API Wrapper
Supports: text generation / function calling (entity extraction) / image understanding
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
    MiniMax API Client
    
    Features:
    1. Text generation (Entity / Relation extraction, classification, quality scoring)
    2. Image understanding (chart description, multimodal understanding)
    3. Function calling (Tool Use)
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
            logger.warning("MINIMAX_API_KEY is not set, some features will be unavailable")
    
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """
        Text generation
        
        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum generation length
            json_mode: Whether to return JSON
            
        Returns:
            Generated text
        """
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY is not set")
        
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
            logger.error(f"MiniMax API call failed: {e}")
            raise
    
    def generate_with_functions(
        self,
        prompt: str,
        functions: list[dict],
        system: str = "",
        temperature: float = 0.3,
    ) -> dict:
        """
        Function calling (Tool Use)
        
        Used for: structured entity extraction, relation extraction
        
        Args:
            prompt: User prompt
            functions: List of function definitions (OpenAI format)
            system: System prompt
            
        Returns:
            Function call result (parsed)
        """
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY is not set")
        
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
            logger.error(f"MiniMax function call failed: {e}")
            raise
    
    def understand_image(
        self,
        image_bytes: bytes,
        prompt: str = "Please describe this image in detail",
        detail: str = "high",
    ) -> str:
        """
        Image understanding (multimodal)
        
        Used for: chart description, scanned document content extraction, complex page understanding
        
        Args:
            image_bytes: Image byte data
            prompt: Query prompt
            detail: low / high (high resolution = more details)
            
        Returns:
            Image description text
        """
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY is not set")
        
        import requests
        
        # Base64 encoding
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
            logger.error(f"MiniMax image understanding failed: {e}")
            raise
    
    def extract_entities(
        self,
        text: str,
        schema: dict,
    ) -> dict:
        """
        Entity + relation extraction (via function calling)
        
        Args:
            text: Text to extract from
            schema: Extraction schema
            
        Returns:
            {"entities": [...], "relations": [...]}
        """
        entity_type = schema.get("entities", [])
        relation_type = schema.get("relations", [])
        
        entity_def = "\n".join([f"- {e}: Entity type" for e in entity_type])
        relation_def = "\n".join([f"- {r}: Relation type" for r in relation_type])
        
        functions = [
            {
                "name": "extract_knowledge",
                "description": "Extract entities and relations from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "description": "List of extracted entities",
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
                            "description": "List of extracted relations",
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
        
        prompt = f"""Please extract entities and relations from the following text.

Entity types:
{entity_def}

Relation types:
{relation_def}

Text:
---
{text[:4000]}
---

Please call the extract_knowledge function to output the results."""
        
        system = "You are a professional knowledge extraction system. Only output JSON, no other content."
        
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
        Text classification (LLM)
        
        Returns:
            {"category": "classification result", "confidence": 0.0-1.0}
        """
        prompt = f"""Please classify the following text into the most appropriate category.

Available categories: {', '.join(categories)}

Text:
---
{text[:2000]}
---

Output JSON:
{{"category": "category name", "confidence": 0.0-1.0}}"""
        
        system = "You are a text classification expert. Only output JSON."
        
        response = self.generate(prompt, system=system, temperature=0.3)
        
        try:
            import re
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                return json.loads(match.group())
        except:
            pass
        
        return {"category": categories[0] if categories else "unknown", "confidence": 0.0}
