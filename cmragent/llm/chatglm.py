from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from zhipuai import ZhipuAI
from .model_name import MODEL_CATEGORIES

class ChatGLM(LLM):
    model: str
    temperature: float = 0.1
    api_key: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        client = ZhipuAI(api_key=self.api_key)

        def gen_chatglm_params(prompt):
            messages = [{"role": "user", "content": prompt}]
            return messages

        messages = gen_chatglm_params(prompt)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        if len(response.choices) > 0:
            return response.choices[0].message.content
        return "generate answer error"

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
        }

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            **self._default_params
        }

def get_chatglm_llm(model_name: str, temperature: float = 0.1, api_key: Optional[str] = None):
    model_name = model_name.lower()

    valid_models = [model.lower() for models in MODEL_CATEGORIES.values() for model in models]

    if model_name not in valid_models:
        raise ValueError(f"Invalid ChatGLM model_name: {model_name}. Must be one of {valid_models}.")

    if not api_key:
        raise ValueError("API key is required for ChatGLM models.")

    llm = ChatGLM(model=model_name, temperature=temperature, api_key=api_key)

    return llm

