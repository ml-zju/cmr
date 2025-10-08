import time
import requests
import json
from langchain.tools import BaseTool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .get_cid import get_compound_cid
from langchain.llms import BaseLLM

class BasePubchemTool(BaseTool):
    llm: BaseLLM = None
    llm_chain: LLMChain = None
    properties: list = []
    raw_data: dict = {}

    def __init__(self, llm: BaseLLM, properties: list, prompt_template: str):
        super().__init__()
        self.llm = llm
        self.properties = properties
        prompt = PromptTemplate(template=prompt_template, input_variables=["data"])
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    def _fetch_pubchem_data(self, identifier: str, max_retries: int = 3) -> dict:
        for attempt in range(max_retries):
            try:
                cid = get_compound_cid(identifier)
                print(f"Found CID: {cid}")
                data = {'CID': cid}
                url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?heading={self.name}'
                req = requests.get(url)
                proper_json = json.loads(req.text)

                try:
                    section = proper_json['Record']['Section'][0]['Section'][0]['Section']
                    all_descriptions = []

                    # 遍历所有部分
                    for sec in section:
                        if 'Information' in sec:
                            for info in sec['Information']:
                                if 'Value' in info and 'StringWithMarkup' in info['Value']:
                                    for markup in info['Value']['StringWithMarkup']:
                                        if 'String' in markup:
                                            all_descriptions.append(markup['String'])

                    data['Description'] = '; '.join(all_descriptions) if all_descriptions else None
                    self.raw_data = proper_json
                    print(f"\nRaw PubChem Data for {identifier}:\n{data['Description']}")
                    return data
                except KeyError:
                    print("Failed to extract description from JSON response")
                    return {'Description': None}
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return {'Description': None}
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching data for {identifier}: {e}")
                return {'Description': None}

    def _run(self, input_str: str) -> str:
        try:
            data = self._fetch_pubchem_data(input_str)
            print("Fetched data:", data)
            return data.get('Description') or 'No information found'
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return f"Error: {str(e)}"

    def _arun(self, input_str: str) -> str:
        raise NotImplementedError("Async not implemented")

    def get(self, input_str: str) -> str:
        return self._run(input_str)


def create_pubchem_tool(property_name: str, property_description: str):
    """
    Factory function to create a PubChem tool for a specific property.

    Args:
        property_name (str): Name of the property to extract
        property_description (str): Description of the property

    Returns:
        class: A BasePubchemTool subclass configured for the specified property
    """

    class PubchemTool(BasePubchemTool):
        name: str = property_name.lower()
        description: str = f"""Tool to find the {property_name} of chemical compounds.
        Input: Chemical name, CAS number, or SMILES notation
        Output: {property_name} value as reported by PubChem

        {property_description}"""

        def __init__(self, llm: BaseLLM):
            prompt_template = f"Extract the {property_name.lower()} value from: {{data}}"
            super().__init__(llm, [property_name], prompt_template)

    return PubchemTool