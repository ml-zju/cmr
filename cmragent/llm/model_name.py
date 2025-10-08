MODEL_CATEGORIES = {
    "DeepSeek": ["deepseek-chat"],
    "ChatGPT": [
            'gpt-5', 'gpt-4o', 'gpt-4o-2024-05-13', 'gpt-4o-mini', 'gpt-4o-mini-2024-07-18',
            'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-instruct', 'gpt-4-turbo',
            'gpt-4-turbo-2024-04-09', 'gpt-4', 'gpt-4-32k', 'gpt-3.5-turbo-0125',
            'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613',
        ],
    "ChatGLM": ['glm-4-plus', 'glm-4', 'glm-3-turbo', 'glm-4-0520', 'glm-4-air', 'glm-4-long'],
    "Spark": ['spark-4.0-ultra', 'spark-3.5-max', 'spark-pro', 'spark-v2.0', 'spark-lite'],
    "ERNIE": [
        'ernie-speed-8k', 'ernie-bot-turbo', 'ernie-4.0-8k-preview', 'ernie-4.0-8k-latest',
        'ernie-3.5-preview', 'ernie-3.5-8k-0205', 'ernie-3.5-8k-0329', 'ernie-3.5-128k',
        'ernie-3.5-8k', 'ernie-3.5-8k-preview', 'ernie-speed-128k', 'ernie-tiny-8k'
    ],
    "Qwen":['qwq-plus', 'qwq-plus-latest', 'qwen-max', 'qwen-max-latest', 'qwen-plus', 'qwen-plus-latest',
            'qwen-turbo', 'qwen-turbo-latest', 'qwen-long', 'qwen-long-latest',
            'qwq-32b', 'qwen2-72b-instruct', 'qwen-72b-chat'],
    "Gemini": [
        'gemini-1.5-flash', 'gemini-1.5-flash-exp-0827', 'gemini-2.0-flash'
        'gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-2.0-flash-thinking-exp'
        'gemini-1.5-pro-exp-0827', 'gemini-1.5-pro-latest', 'gemini-2.0-pro-exp'
    ],
}

SPARK_MODELS = {
    'spark-4.0-ultra': ('wss://spark-api.xf-yun.com/v4.0/chat', '4.0Ultra'),
    'spark-3.5-max': ('wss://spark-api.xf-yun.com/v3.5/chat', 'generalv3.5'),
    'spark-pro': ('wss://spark-api.xf-yun.com/v3.1/chat', 'generalv3'),
    'spark-v2.0': ('wss://spark-api.xf-yun.com/v2.1/chat', 'generalv2'),
    'spark-lite': ('wss://spark-api.xf-yun.com/v1.1/chat', 'general'),
}