from typing import Any, List, Optional, Iterator
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI
from pydantic import Field, PrivateAttr
import os
from .model_name import MODEL_CATEGORIES

WENXIN_MODEL = MODEL_CATEGORIES["Qwen"]

class QwenLLM(LLM):
    model: str = Field(..., description="The model name to use")
    api_key: str = Field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY"))
    temperature: float = Field(default=0.1)
    base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    _client: Optional[OpenAI] = PrivateAttr(default=None)

    @property
    def client(self) -> OpenAI:
        if not self._client:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> str:
        try:
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stop=stop,
                stream=True,
                **kwargs
            )

            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.choices[0].delta.content)

            return full_response
        except Exception as e:
            raise RuntimeError(f"Error calling Qwen API: {str(e)}") from e

    @property
    def _llm_type(self) -> str:
        return "qwen"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model, "temperature": self.temperature, "base_url": self.base_url}


def get_qwen_llm(
        model_name: str,
        api_key: str = None,
        temperature: float = 0.1,
        **kwargs: Any
) -> QwenLLM:
    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("API key is required. Set DASHSCOPE_API_KEY environment variable or provide api_key.")

    if model_name.lower() not in [model.lower() for model in WENXIN_MODEL]:
        raise ValueError(f"Invalid model name: {model_name}. Valid models are: {', '.join(WENXIN_MODEL)}")

    return QwenLLM(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        **kwargs
    )