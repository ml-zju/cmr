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


class ReproductiveToxicityEvidenceTool(BaseTool):
    name: str = "ReproductiveToxicityEvidenceTool"
    description: str = (
        "Input chemical name, CAS number, or SMILES notation, returns a comprehensive summary of reproductive toxicity evidence. "
        "The summary includes household product information, hazard classifications, long-term exposure effects, "
        "regulatory information, toxicity data, and reproductive health-related assessments."
    )

    llm: BaseLLM = None
    properties: list = []
    headings: List[str] = [
        "Household Products",
        "Hazard Classes and Categories",
        "Effects of Long Term Exposure",
        "Regulatory Information",
        "Special Reports",
        "Toxicity Summary",
        "Interactions",
        "National Toxicology Program Studies",
        "Ecotoxicity Excerpts",
        "Hazards Summary",
    ]

    def __init__(self, llm: BaseLLM):
        super().__init__()
        self.llm = llm
        self.properties = ["Reproductive Toxicity Evidence"]

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

    def get_reproductive_toxicity_evidence_summary(self, identifier: str) -> str:
        all_data = []
        for heading in self.headings:
            data = self._fetch_pubchem_data(identifier, heading)
            if data['Description'] is not None:
                all_data.append(f"{heading}: {data['Description']}")

        if not all_data:
            return "No reproductive toxicity evidence data found."

        reproductive_toxicity_data = "\n\n".join(all_data)

        prompt_template = PromptTemplate(
            template="""
                Please review the following reproductive toxicity evidence data and provide a clear and concise summary based on the available information. 
                Organize the summary according to each reproductive toxicity evidence category, focusing on:
                - Effects on fertility and reproductive function
                - Developmental toxicity and teratogenic effects
                - Endocrine disruption potential
                - Pregnancy and lactation safety
                - Reproductive organ toxicity
                - Regulatory classifications for reproductive hazards
                - Long-term exposure effects on reproductive health
                - National Toxicology Program findings related to reproduction

                Here is the data:
                {data}
                """,
            input_variables=["data"]
        )
        llm_chain = LLMChain(prompt=prompt_template, llm=self.llm)
        summary = llm_chain.run({"data": reproductive_toxicity_data})
        return summary

    def _run(self, input_str: str) -> str:
        try:
            summary = self.get_reproductive_toxicity_evidence_summary(input_str)
            return summary
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return f"Error: {str(e)}"

    def _arun(self, query: str) -> str:
        raise NotImplementedError("ReproductiveToxicityEvidenceTool does not support async")