from typing import Any, Dict, List, Mapping, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI
from .model_name import MODEL_CATEGORIES
from pydantic import Field, PrivateAttr


class DeepseekLLM(LLM):
    """Deepseek Language Model wrapper for LangChain."""

    model: str = Field(..., description="The model name to use")
    api_key: str = Field(..., description="Deepseek API key")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    base_url: str = Field(default="https://api.deepseek.com", description="API base URL")

    _client: Optional[OpenAI] = PrivateAttr(default=None)

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> str:
        """Execute the LLM call."""
        try:
            messages = [{"role": "user", "content": prompt}]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stop=stop,
                **kwargs
            )

            if not response.choices:
                raise ValueError("No response generated from the model")

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error calling Deepseek API: {str(e)}") from e

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "deepseek"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "base_url": self.base_url
        }


def get_deepseek_llm(
        model_name: str,
        api_key: str,
        temperature: float = 0.1,
        **kwargs: Any
) -> DeepseekLLM:
    if not api_key:
        raise ValueError("API key is required")

    model_name = model_name.lower()
    valid_models = {
        model.lower()
        for models in MODEL_CATEGORIES.values()
        for model in models
    }

    if model_name not in valid_models:
        raise ValueError(
            f"Invalid model name: {model_name}. "
            f"Valid models are: {', '.join(sorted(valid_models))}"
        )

    return DeepseekLLM(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        **kwargs
    )