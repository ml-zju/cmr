import re
from typing import Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser

class CustomOutputParser(AgentOutputParser):
    def __init__(self):
        super().__init__()
        self._last_action = None

    @property
    def last_action(self):
        return self._last_action

    @last_action.setter
    def last_action(self, value):
        self._last_action = value

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Observation:" in text:
            observation = text.split("Observation:")[-1].strip()
            if observation and len(observation) > 10:
                return AgentFinish(
                    return_values={"output": observation},
                    log=text
                )

        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )

        action_match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*)", text, re.DOTALL)
        if not action_match:
            return AgentFinish(
                return_values={"output": text.strip()},
                log=text
            )

        action = action_match.group(1).strip()
        action_input = action_match.group(2).strip()

        if self._last_action == (action, action_input):
            return AgentFinish(
                return_values={"output": "Previous observation is the final answer."},
                log=text
            )

        self._last_action = (action, action_input)
        return AgentAction(tool=action, tool_input=action_input, log=text)