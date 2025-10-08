import re
from rdkit import Chem
import pubchempy as pcp
from langchain.tools import BaseTool, Tool
from pydantic import BaseModel, Field
from typing import Type
from tools.get_cid import get_compound_cid
import requests

def is_valid_smiles(smiles_string):
    try:
        molecule = Chem.MolFromSmiles(smiles_string, sanitize=False)
        return molecule is not None
    except:
        return False

def is_valid_cas(cas_string):
    pattern = r"^\d{2,7}-\d{2}-\d$"
    return re.match(pattern, cas_string) is not None

def get_canonical_smiles(smiles_string):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles_string), canonical=True)
    except Exception:
        return "Invalid SMILES string"

def get_largest_molecule(smiles_string):
    components = smiles_string.split(".")
    components.sort(key=lambda x: len(x))
    while components and not is_valid_smiles(components[-1]):
        components.pop()
    return components[-1] if components else ""

def get_cas_from_pubchem(query):
    try:
        cid = get_compound_cid(query)
        if not cid:
            return "No CID found"

        compound = pcp.Compound.from_cid(cid)
        cas = None

        for synonym in compound.synonyms:
            if is_valid_cas(synonym):
                cas = synonym
                break

        return cas if cas else "CAS number not found"

    except Exception as e:
        return f"Error retrieving CAS: {str(e)}"

def get_name_from_pubchem(identifier):
    try:
        cid = get_compound_cid(identifier)
        if not cid:
            return "No CID found"

        compound = pcp.Compound.from_cid(cid)
        return compound.iupac_name

    except Exception as e:
        return f"Error retrieving name: {str(e)}"


def get_smiles_from_pubchem(identifier):
    try:
        cid = get_compound_cid(identifier)
        if not cid:
            return "No CID found"

        try:
            compound = pcp.Compound.from_cid(cid)
            if hasattr(compound, 'isomeric_smiles') and compound.isomeric_smiles:
                return compound.isomeric_smiles
            elif hasattr(compound, 'canonical_smiles') and compound.canonical_smiles:
                return compound.canonical_smiles
        except:
            pass

        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['IsomericSMILES']
                return smiles
        except:
            pass

        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                return smiles
        except:
            pass

        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?heading=SMILES"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()

                def find_smiles_in_sections(sections):
                    for section in sections:
                        if 'Information' in section:
                            for info in section['Information']:
                                if 'Value' in info:
                                    if 'StringWithMarkup' in info['Value']:
                                        for markup in info['Value']['StringWithMarkup']:
                                            if 'String' in markup:
                                                potential_smiles = markup['String']
                                                if len(potential_smiles) > 5 and any(
                                                        c in potential_smiles for c in 'CNOPSFcnops()[]=#'):
                                                    return potential_smiles

                                    elif 'String' in info['Value']:
                                        potential_smiles = info['Value']['String']
                                        if len(potential_smiles) > 5 and any(
                                                c in potential_smiles for c in 'CNOPSFcnops()[]=#'):
                                            return potential_smiles

                        if 'Section' in section:
                            result = find_smiles_in_sections(section['Section'])
                            if result:
                                return result
                    return None

                if 'Record' in data and 'Section' in data['Record']:
                    smiles = find_smiles_in_sections(data['Record']['Section'])
                    if smiles:
                        return smiles
        except:
            pass

        return "SMILES not found"

    except Exception as e:
        return f"Error retrieving SMILES: {str(e)}"

def get_inchi_from_pubchem(identifier):
    try:
        cid = get_compound_cid(identifier)
        if not cid:
            return "No CID found"

        compound = pcp.Compound.from_cid(cid)
        return compound.inchi

    except Exception as e:
        return f"Error retrieving InChI: {str(e)}"

def get_inchikey_from_pubchem(identifier):
    try:
        cid = get_compound_cid(identifier)
        if not cid:
            return "No CID found"

        compound = pcp.Compound.from_cid(cid)
        return compound.inchikey

    except Exception as e:
        return f"Error retrieving InChIKey: {str(e)}"

def get_formula_from_pubchem(identifier):
    try:
        cid = get_compound_cid(identifier)
        if not cid:
            return "No CID found"

        compound = pcp.Compound.from_cid(cid)
        return compound.molecular_formula

    except Exception as e:
        return f"Error retrieving molecular formula: {str(e)}"

class ChemToolInput(BaseModel):
    query: str = Field(..., description="The input query string")

class ChemToolOutput(BaseModel):
    result: str = Field(..., description="The result of the chemical tool operation")

class CASRetriever(BaseTool):
    name: str = "CASRetriever"
    description: str = "Input a molecule name or SMILES string to retrieve the CAS number."
    args_schema: Type[BaseModel] = ChemToolInput

    def _run(self, query: str) -> str:
        return get_cas_from_pubchem(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError()

    def get(self, query: str) -> str:
        return self._run(query)

class SMILESRetriever(BaseTool):
    name: str = "SMILESRetriever"
    description: str = "Input a CAS number or name to retrieve the canonical SMILES string."
    args_schema: Type[BaseModel] = ChemToolInput

    def _run(self, query: str) -> str:
        return get_smiles_from_pubchem(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError()

    def get(self, query: str) -> str:
        return self._run(query)

class NameRetriever(BaseTool):
    name: str = "NameRetriever"
    description: str = "Input a CAS number or SMILES string to retrieve the molecule name."
    args_schema: Type[BaseModel] = ChemToolInput

    def _run(self, query: str) -> str:
        if is_valid_cas(query):
            return get_name_from_pubchem(query)
        return get_name_from_pubchem(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError()

    def get(self, query: str) -> str:
        return self._run(query)

class InChIRetriever(BaseTool):
    name: str = "InChIRetriever"
    description: str = "Input a CAS number, name or SMILES string to retrieve the InChI."
    args_schema: Type[BaseModel] = ChemToolInput

    def _run(self, query: str) -> str:
        return get_inchi_from_pubchem(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError()

    def get(self, query: str) -> str:
        return self._run(query)

class InChIKeyRetriever(BaseTool):
    name: str = "InChIKeyRetriever"
    description: str = "Input a CAS number, name or SMILES string to retrieve the InChIKey."
    args_schema: Type[BaseModel] = ChemToolInput

    def _run(self, query: str) -> str:
        return get_inchikey_from_pubchem(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError()

    def get(self, query: str) -> str:
        return self._run(query)

class FormulaRetriever(BaseTool):
    name: str = "FormulaRetriever"
    description: str = "Input a CAS number, name or SMILES string to retrieve the molecular formula."
    args_schema: Type[BaseModel] = ChemToolInput

    def _run(self, query: str) -> str:
        return get_formula_from_pubchem(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError()

    def get(self, query: str) -> str:
        return self._run(query)

def create_chemical_tools():
    return [
        Tool(
            name="CASRetriever",
            func=CASRetriever().get,
            description="Input a molecule name or SMILES string to retrieve the CAS number."
        ),
        Tool(
            name="SMILESRetriever",
            func=SMILESRetriever().get,
            description="Input a CAS number or name to retrieve the canonical SMILES string."
        ),
        Tool(
            name="NameRetriever",
            func=NameRetriever().get,
            description="Input a CAS number or SMILES string to retrieve the molecule name."
        ),
        Tool(
            name="InChIRetriever",
            func=InChIRetriever().get,
            description="Input a CAS number, name or SMILES string to retrieve the InChI."
        ),
        Tool(
            name="InChIKeyRetriever",
            func=InChIKeyRetriever().get,
            description="Input a CAS number, name or SMILES string to retrieve the InChIKey."
        ),
        Tool(
            name="FormulaRetriever",
            func=FormulaRetriever().get,
            description="Input a CAS number, name or SMILES string to retrieve the molecular formula."
        ),
    ]


