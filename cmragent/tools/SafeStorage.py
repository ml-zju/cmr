import re
import json
import time
import requests
from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from langchain.tools import BaseTool
from rdkit import Chem
from typing import List
from .get_cid import get_compound_cid

def is_smiles(text):
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        if m is None:
            return False
        return True
    except:
        return False

def is_cas(text):
    pattern = r"^\d{2,7}-\d{2}-\d$"
    return re.match(pattern, text) is not None

class SafeStorageTool(BaseTool):
    name: str = "SafeStorageTool"
    description: str = (
        "Input chemical name, CAS number, or SMILES notation, returns a summary of safe storage information for the chemical. "
        "The summary includes safe storage measures, storage conditions, incompatible chemicals, and any specific storage requirements."
    )

    llm: BaseLLM = None
    properties: list = []
    headings: List[str] = [
        "Safe storage",
        "Storage conditions",
    ]

    def __init__(self, llm: BaseLLM):
        super().__init__()
        self.llm = llm
        self.properties = ["Safe Storage"]

    def _fetch_pubchem_data(self, identifier: str, heading: str, max_retries: int = 3) -> dict:
        for attempt in range(max_retries):
            try:
                cid = get_compound_cid(identifier)
                print(f"Found CID: {cid}")
                data = {'CID': cid}
                url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?heading={heading}'
                req = requests.get(url)
                proper_json = json.loads(req.text)

                try:
                    Section = proper_json['Record']['Section'][0]['Section'][0]['Section']
                    value = Section[0]['Information'][0]['Value']['StringWithMarkup'][0]['String']
                    data['Description'] = value
                    print(f"\nRaw PubChem Data for {identifier} ({heading}):")
                    print(f"{value}")
                    return data
                except KeyError:
                    print(f"Failed to extract description from JSON response for {heading}")
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

    def get_safe_storage_summary(self, identifier: str) -> str:
        all_data = []
        for heading in self.headings:
            data = self._fetch_pubchem_data(identifier, heading)
            if data['Description'] is not None:
                all_data.append(f"{heading}: {data['Description']}")

        if not all_data:
            return "No safe storage data found."

        storage_data = "\n\n".join(all_data)

        prompt_template = PromptTemplate(
            template="""
                Please review the following information about safe storage of this chemical. 
                Summarize the safe storage information based on the available content. 

                Here is the data:
                {data}
                """,
            input_variables=["data"]
        )
        llm_chain = LLMChain(prompt=prompt_template, llm=self.llm)
        summary = llm_chain.run({"data": storage_data})
        return summary

    def _run(self, input_str: str) -> str:
        try:
            summary = self.get_safe_storage_summary(input_str)
            return summary
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return f"Error: {str(e)}"

    def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SafeStorageTool does not support async")