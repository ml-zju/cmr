import re
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction

class ThoughtCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.current_process = []
        self.complete_process = []
        self.has_final_answer = False

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
        self.current_process = []
        self.has_final_answer = False
        self.current_process.append("> Entering new AgentExecutor chain...")

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        thought_match = re.search(r"Thought:(.*?)(?=Action:|$)", action.log, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
            self.current_process.append(f"Thought: {thought}")

        self.current_process.append(f"Action: {action.tool}")
        self.current_process.append(f"Action Input: {action.tool_input}")

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        if "Searching for CID of:" in input_str:
            self.current_process.append(input_str)

    def on_tool_end(self, output: str, **kwargs) -> None:
        lines = output.split('\n')
        for line in lines:
            if "Found CID:" in line:
                self.current_process.append(line)
            elif "Raw PubChem Data for" in line:
                self.current_process.append(line)
            elif "Fetched data:" in line:
                self.current_process.append(line)

        if not output.startswith("Observation:"):
            self.current_process.append(f"\nObservation: {output}")
        else:
            self.current_process.append(f"\n{output}")

    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        if "output" in outputs:
            self.current_process.append(f"Final Answer: {outputs['output']}")
            self.current_process.append("\n> Finished chain.")
            self.has_final_answer = True
            self.complete_process = self.current_process.copy()

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        self.current_process.append(f"\nError: {str(error)}")

    def clear(self):
        self.current_process = []
        self.complete_process = []
        self.has_final_answer = False

    def get_thoughts(self):
        if self.has_final_answer and self.complete_process:
            return "\n".join(self.complete_process)
        return "No complete process available yet."