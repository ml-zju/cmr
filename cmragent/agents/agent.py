import logging
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import LLMChain
from langchain.tools import Tool
from .prompts import (
    FORMAT_INSTRUCTIONS,
    QUESTION_PROMPT,
    SUFFIX,
    PREFIX,
    REPHRASE_TEMPLATE,
    CustomPromptTemplate
)
from .output_parser import CustomOutputParser
from .callback_handler import ThoughtCallbackHandler


class CMR:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.agent = None
        self.logger = None
        self.callback_handler = ThoughtCallbackHandler()
        self.setup_logging()
        self.create_agent()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def create_agent(self):
        prompt = CustomPromptTemplate(
            template=f"{PREFIX}\n\n{FORMAT_INSTRUCTIONS}\n\n{QUESTION_PROMPT}\n\n{SUFFIX}",
            tools=self.tools,
            input_variables=["input", "intermediate_steps", "agent_scratchpad"]
        )

        output_parser = CustomOutputParser()

        agent = LLMSingleActionAgent(
            llm_chain=LLMChain(
                llm=self.llm,
                prompt=prompt
            ),
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )

        self.agent = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=6,
            handle_parsing_errors=True,
            callbacks=[self.callback_handler]
        )

    def run(self, input: str):
        if not self.agent:
            raise ValueError("Agent has not been initialized")

        self.callback_handler.clear()

        try:
            result = self.agent.run(
                input=input,
                agent_scratchpad=""
            )
            return {
                "answer": result,
                "thought_process": self.callback_handler.get_thoughts()
            }
        except Exception as e:
            self.logger.error(f"Error during agent execution: {str(e)}")
            return {
                "answer": "Error occurred during processing",
                "thought_process": self.callback_handler.get_thoughts(),
                "error": str(e)
            }

