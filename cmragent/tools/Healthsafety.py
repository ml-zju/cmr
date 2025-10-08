import os
import re
import requests
import json
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from rdkit import Chem
from typing import Optional

def is_cas(text):
    pattern = r"^\d{2,7}-\d{2}-\d$"
    return re.match(pattern, text) is not None

def is_smiles(text):
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        return m is not None
    except Exception:
        return False


class MoleculeSafety:
    def __init__(self):
        self.pubchem_data = {}

    def _fetch_hazard_info(self, cid, heading):
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?heading={heading}'
        try:
            req = requests.get(url)
            proper_json = json.loads(req.text)
            section = proper_json['Record']['Section'][0]['Section'][0]['Section']
            info = []
            for item in section:
                if 'Information' in item:
                    for info_item in item['Information']:
                        if 'Value' in info_item and 'StringWithMarkup' in info_item['Value']:
                            info.append(info_item['Value']['StringWithMarkup'][0]['String'])
            return info
        except Exception as e:
            return None

    def _fetch_pubchem_cid(self, identifier):
        try:
            if is_cas(identifier):
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{identifier}/cids/JSON"
            elif is_smiles(identifier):
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{identifier}/cids/JSON"
            else:
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{identifier}/cids/JSON"

            response = requests.get(url)
            response.raise_for_status()
            cid = response.json()['IdentifierList']['CID'][0]
            return cid
        except Exception as e:
            return None

class HealthSafetyTool(BaseTool):
    name: str = "HealthHazards"
    description: str = (
        "Input CAS number, SMILES, or chemical name to get information about health effects from PubChem. "
        "The tool will first determine the type of input (CAS, SMILES, or name) and then use the appropriate URL "
        "to search for the health effects data. It can retrieve data on Explosion Hazards, Fire Hazards, Fire Potential, "
        "Hazards Summary, Health Hazards, and Skin, Eye, and Respiratory Irritations."
    )

    mol_safety: Optional[MoleculeSafety] = None
    llm: Optional[BaseLLM] = None
    llm_chain: Optional[LLMChain] = None

    def __init__(self, llm: BaseLLM):
        super().__init__()
        self.mol_safety = MoleculeSafety()
        self.llm = llm
        prompt = PromptTemplate(
            template=(
                "Based on the following hazard information, provide a concise and clear summary:\n\n"
                "Health Hazards:\n{health_hazards}\n\n"
                "Hazard Classes and Categories:\n{hazard_classes}\n\n"
                "Fire Hazards:\n{fire_hazards}\n\n"
                "Hazards Summary:\n{hazards_summary}\n\n"
                "Skin, Eye, and Respiratory Irritations:\n{irritations}\n\n"
                "Summary:"
            ),
            input_variables=["health_hazards", "hazard_classes", "fire_hazards", "hazards_summary", "irritations"]
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    def _run(self, identifier: str) -> str:
        # 确保 mol_safety 属性存在
        if not hasattr(self, "mol_safety") or self.mol_safety is None:
            self.mol_safety = MoleculeSafety()

        cid = self.mol_safety._fetch_pubchem_cid(identifier)
        if not cid:
            return "Molecule not found in PubChem."

        health_hazards = self.mol_safety._fetch_hazard_info(cid, "Health Hazards")
        hazard_classes = self.mol_safety._fetch_hazard_info(cid, "Hazard Classes and Categories")
        fire_hazards = self.mol_safety._fetch_hazard_info(cid, "Fire Hazards")
        hazards_summary = self.mol_safety._fetch_hazard_info(cid, "Hazards Summary")
        irritations = self.mol_safety._fetch_hazard_info(cid, "Skin, Eye, and Respiratory Irritations")

        health_hazards_str = "\n".join(health_hazards) if health_hazards else "No information available."
        hazard_classes_str = "\n".join(hazard_classes) if hazard_classes else "No information available."
        fire_hazards_str = "\n".join(fire_hazards) if fire_hazards else "No information available."
        hazards_summary_str = "\n".join(hazards_summary) if hazards_summary else "No information available."
        irritations_str = "\n".join(irritations) if irritations else "No information available."

        summary = self.llm_chain.run({
            "health_hazards": health_hazards_str,
            "hazard_classes": hazard_classes_str,
            "fire_hazards": fire_hazards_str,
            "hazards_summary": hazards_summary_str,
            "irritations": irritations_str
        })

        return summary

    async def _arun(self, identifier: str) -> str:
        raise NotImplementedError("Async execution is not supported for this tool.")