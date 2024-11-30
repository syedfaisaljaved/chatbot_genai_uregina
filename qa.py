# qa.py
from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.components.model_client import OllamaClient
import logging
from typing import Dict, Any

from config import Config


class OllamaQA(Component):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.generator = Generator(
            model_client=OllamaClient(),
            model_kwargs={"model": config.model_name},
            template=r"""<SYS>
You are the official AI assistant for the University of Regina. Answer questions directly and confidently.

Rules:
1. Never mention "context", "data", or make references to your information source
2. If information isn't in your knowledge base, respond: "I don't have that specific information about the University of Regina"
3. Focus only on official university information
4. No personal opinions or advice
5. No speculation or assumptions

Context: {{context}}
</SYS>
User: {{question}}
Assistant:
""")

    def get_response(self, context: str, question: str) -> Dict[str, Any]:
        try:
            if not context.strip():
                return {
                    "answer": "I don't have that specific information about the University of Regina.",
                    "success": True
                }

            response = self.generator.call({
                "context": context,
                "question": question
            })
            return {"answer": response.data, "success": True}
        except Exception as e:
            logging.error(f"Error getting response from Ollama: {e}")
            return {
                "answer": "Sorry, I encountered an error processing your question.",
                "success": False
            }
