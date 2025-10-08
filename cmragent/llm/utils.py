from llm.chatgpt import get_openai_llm
from llm.chatglm import get_chatglm_llm
from llm.spark import get_spark_llm
from llm.wenxin import get_wenxin_llm
from llm.deepseek import get_deepseek_llm
from llm.Qwen import get_qwen_llm
from langchain.base_language import BaseLanguageModel
from typing import Optional
from .model_name import MODEL_CATEGORIES
from llm.gemini import get_gemini_llm

def get_llm(model_name: str, temperature: float = 0.1, max_tokens: int = 4096,
            api_key: Optional[str] = None, secret_key: Optional[str] = None,
            system: Optional[str] = None, token: Optional[str] = None,
            appid: Optional[str] = None, api_secret: Optional[str] = None,
            spark_url: Optional[str] = None) -> BaseLanguageModel:
    m_name = model_name.lower()

    model_category = None
    for category, models in MODEL_CATEGORIES.items():
        if m_name in [m.lower() for m in models]:
            model_category = category
            break

    if not model_category:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported models are {', '.join(MODEL_CATEGORIES.keys())}.")

    if model_category == "DeepSeek":
        if api_key is None:
            raise ValueError("API key is required for DeepSeek models.")
        llm = get_deepseek_llm(model_name, temperature=temperature, api_key=api_key)
    elif model_category == "Gemini":
        if api_key is None:
            raise ValueError("API key is required for GPT models.")
        llm = get_gemini_llm(model_name, temperature=temperature, api_key=api_key)
    elif model_category == "GPT":
        if api_key is None:
            raise ValueError("API key is required for GPT models.")
        llm = get_openai_llm(model_name, temperature=temperature, api_key=api_key)
    elif model_category == "ChatGLM":
        if api_key is None:
            raise ValueError("API key is required for ChatGLM models.")
        llm = get_chatglm_llm(model_name, temperature=temperature, api_key=api_key)
    elif model_category == "Spark":
        if not all([appid, api_key, api_secret]):
            raise ValueError("App ID, API key, and API secret are required for Spark models.")
        llm = get_spark_llm(model_name, temperature=temperature, appid=appid, api_key=api_key, api_secret=api_secret)
    elif model_category == "Qwen":
        if api_key is None:
            raise ValueError("API key is required for Qwen models.")
        llm = get_qwen_llm(model_name, temperature=temperature, api_key=api_key)
    elif model_category == "ERNIE":
        if api_key is None:
            raise ValueError("API key is required for ERNIE models.")
        llm = get_wenxin_llm(model_name, temperature=temperature, api_key=api_key, secret_key=secret_key)
    else:
        raise ValueError(f"Unsupported model_category: {model_category}. Supported categories are {', '.join(MODEL_CATEGORIES.keys())}.")

    return llm






