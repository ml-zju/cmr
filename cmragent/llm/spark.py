from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from sparkai.llm.llm import ChatSparkLLM
from sparkai.core.messages import ChatMessage
from .model_name import SPARK_MODELS

class ChatSpark(LLM):
    model: str
    temperature: float = 0.1
    appid: str
    api_key: str
    api_secret: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        spark_params = self._get_spark_params(self.model)

        spark_llm = ChatSparkLLM(
            spark_api_url=spark_params["spark_url"],
            spark_app_id=self.appid,
            spark_api_key=self.api_key,
            spark_api_secret=self.api_secret,
            spark_llm_domain=spark_params["domain"],
            temperature=self.temperature,
            streaming=False
        )

        messages = self._gen_spark_messages(prompt)

        response = spark_llm.generate([messages])

        response_text = response.generations[0][0].text
        return response_text

    @staticmethod
    def _get_spark_params(model: str) -> Dict[str, str]:
        spark_url_tpl = "wss://spark-api.xf-yun.com/{}/chat"
        model_params_dict = {
            "spark-4.0-ultra": {
                "domain": "4.0Ultra",
                "spark_url": spark_url_tpl.format("v4.0")
            },
            "spark-3.5-max": {
                "domain": "generalv3.5",
                "spark_url": spark_url_tpl.format("v3.5")
            },
            "spark-pro": {
                "domain": "generalv3",
                "spark_url": spark_url_tpl.format("v3.1")
            },
            "spark-v2.0": {
                "domain": "generalv2.1",
                "spark_url": spark_url_tpl.format("v2.1")
            },
            "spark-lite": {
                "domain": "general",
                "spark_url": spark_url_tpl.format("v1.1")
            }
        }
        if model not in model_params_dict:
            raise ValueError(f"Model '{model}' is not supported.")
        return model_params_dict[model]

    @staticmethod
    def _gen_spark_messages(prompt: str) -> List[ChatMessage]:
        messages = [ChatMessage(role="user", content=prompt)]
        return messages

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Gets the default parameters for API calls."""
        return {
            "temperature": self.temperature,
        }

    @property
    def _llm_type(self) -> str:
        return "ChatSpark"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Gets the identifying parameters."""
        return {
            "model": self.model,
            **self._default_params
        }


def get_spark_llm(model_name: str, appid: str, api_key: str, api_secret: str,
                       temperature: float = 0.1) -> ChatSpark:
    model_name = model_name.lower()

    valid_models = set(SPARK_MODELS)

    if model_name not in valid_models:
        raise ValueError(f"Invalid ChatSpark model_name: {model_name}. Must be one of {valid_models}.")

    llm = ChatSpark(model=model_name, temperature=temperature, appid=appid, api_key=api_key, api_secret=api_secret)

    return llm
