from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain.base_language import BaseLanguageModel
import qianfan
from .model_name import MODEL_CATEGORIES

WENXIN_MODEL = MODEL_CATEGORIES["ERNIE"]

class Wenxin_LLM(LLM):
    model: str
    temperature: float = 0.1
    api_key: str = None
    secret_key: str = None
    system: str = None

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        def gen_wenxin_messages(prompt):

            messages = [{"role": "user", "content": prompt}]
            return messages

        chat_comp = qianfan.ChatCompletion(ak=self.api_key, sk=self.secret_key)
        message = gen_wenxin_messages(prompt)

        try:
            resp = chat_comp.do(messages=message,
                                model=self.model,
                                temperature=self.temperature,
                                system=self.system)
            return resp["result"]
        except Exception as e:
            raise RuntimeError(f"Failed to get response from Wenxin API: {e}")

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
        }

    @property
    def _llm_type(self) -> str:
        return "Wenxin"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters."""
        return {
            "model": self.model,
            **self._default_params
        }

def get_wenxin_llm(model_name: str, temperature: float = 0.1, api_key: Optional[str] = None,
                   secret_key: Optional[str] = None) -> BaseLanguageModel:
    model_name = model_name.lower()
    if model_name not in WENXIN_MODEL:
        raise ValueError(f"Invalid Wenxin model_name: {model_name}. Must be one of {WENXIN_MODEL}.")

    if not api_key:
        raise ValueError("API key is required for Wenxin models.")

    if secret_key is None:
        raise ValueError("Secret key is required for Wenxin models.")

    llm = Wenxin_LLM(model=model_name, temperature=temperature, api_key=api_key,
                     secret_key=secret_key)

    return llm


