from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.base_language import BaseLanguageModel
from .model_name import MODEL_CATEGORIES

def get_openai_llm(model_name: str, temperature: float = 0.1, api_key: str = None) -> BaseLanguageModel:
    model_name = model_name.lower()

    chat_models = MODEL_CATEGORIES["GPT"]

    if model_name in chat_models:
        if api_key is None:
            raise ValueError("API key is required for ChatOpenAI models")
        llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key)
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

    return llm

