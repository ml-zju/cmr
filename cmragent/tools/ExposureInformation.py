import requests
import re
from rdkit import Chem
from langchain.llms import BaseLLM
from langchain.tools import BaseTool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class ExposureInformation(BaseTool):
    name: str = "ExposureInformation"
    description: str = (
        "Input CAS number, SMILES, or chemical name to get comprehensive exposure-related information from PubChem. "
        "The tool will first determine the type of input (CAS, SMILES, or name) and then use the appropriate URL "
        "to fetch the exposure data."
    )

    llm: BaseLLM = None
    llm_chain: LLMChain = None
    pubchem_data: dict = dict()

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        prompt = PromptTemplate(
            template=(
                "Extract and summarize exposure-related information from the data: {data}. "
                "Include the following sections if available: Animal Concentrations, Artificial Pollution Sources, "
                "Atmospheric Concentrations, Average Daily Intake, Body Burden, Ecotoxicity Values, "
                "Environmental Fate/Exposure Summary, Environmental Water Concentrations, Fish/Seafood Concentrations, "
                "Food Survey Values, Milk Concentrations, Natural Pollution Sources, Other Environmental Concentrations, "
                "Plant Concentrations, Probable Routes of Human Exposure, Sediment/Soil Concentrations, "
                "Volatilization from Water/Soil."
            ),
            input_variables=["data"]
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    @staticmethod
    def is_valid_smiles(smiles_string):
        try:
            molecule = Chem.MolFromSmiles(smiles_string, sanitize=False)
            return molecule is not None
        except Exception as e:
            print(f"Error validating SMILES: {e}")
            return False

    @staticmethod
    def is_valid_cas(cas_string):
        pattern = r"^\d{2,7}-\d{2}-\d$"
        return re.match(pattern, cas_string) is not None

    def _determine_input_type(self, input_str):
        if self.is_valid_cas(input_str):
            return "cas"
        elif self.is_valid_smiles(input_str):
            return "smiles"
        else:
            return "name"

    def _fetch_pubchem_data(self, input_str):
        input_type = self._determine_input_type(input_str)
        if input_str not in self.pubchem_data:
            try:
                if input_type == "cas":
                    url1 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{input_str}/cids/JSON"
                elif input_type == "smiles":
                    url1 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{input_str}/cids/JSON"
                elif input_type == "name":
                    url1 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{input_str}/cids/JSON"
                else:
                    return "Invalid input type. Use 'cas', 'smiles', or 'name'."

                print(f"Fetching data from: {url1}")  # Debug information
                response1 = requests.get(url1)
                response1.raise_for_status()

                cid = response1.json().get('IdentifierList', {}).get('CID', [None])[0]
                if cid is None:
                    return "Invalid molecule input, no PubChem entry."

                url2 = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?heading=Exposure Information'
                print(f"Fetching detailed data from: {url2}")  # Debug information
                response2 = requests.get(url2)
                response2.raise_for_status()

                self.pubchem_data[input_str] = response2.json()
            except requests.exceptions.RequestException as e:
                print(f"Error fetching PubChem data for {input_str}: {e}")
                return "Error fetching data from PubChem."
            except KeyError as e:
                print(f"Error parsing PubChem response for {input_str}: {e}")
                return "Error in response format from PubChem."

        return self.pubchem_data[input_str]

    def _extract_exposure_info(self, section):
        if isinstance(section, list):
            for sec in section:
                result = self._extract_exposure_info(sec)
                if result:
                    return result
        elif isinstance(section, dict):
            if section.get("TOCHeading") in [
                "Animal Concentrations", "Artificial Pollution Sources", "Atmospheric Concentrations",
                "Average Daily Intake", "Body Burden", "Ecotoxicity Values", "Environmental Fate/Exposure Summary",
                "Environmental Water Concentrations", "Fish/Seafood Concentrations", "Food Survey Values",
                "Milk Concentrations", "Natural Pollution Sources", "Other Environmental Concentrations",
                "Plant Concentrations", "Probable Routes of Human Exposure", "Sediment/Soil Concentrations",
                "Volatilization from Water/Soil"
            ]:
                info_list = section.get("Information", [])
                if info_list:
                    return "\n".join(
                        info.get("Value", {}).get("StringWithMarkup", [{}])[0].get("String", "No description available")
                        for info in info_list
                        if "Value" in info and "StringWithMarkup" in info["Value"]
                    )
            if "Section" in section:
                return self._extract_exposure_info(section["Section"])
        return None

    def _run(self, input_str: str) -> str:
        print(f"Running ExposureInformationTool with input: {input_str}")  # Debug information
        data = self._fetch_pubchem_data(input_str)
        if isinstance(data, str):
            return data

        try:
            print(f"Data fetched from PubChem: {data}")  # Debug information
            exposure_section = data.get("Record", {}).get("Section", [])
            if not exposure_section:
                return "No exposure information found."

            result = self._extract_exposure_info(exposure_section)
            return result if result else "No exposure information found."

        except KeyError as e:
            print(f"Error parsing the exposure data: {e}")
            return "Error parsing the exposure data."

    async def _arun(self, input_str: str):
        raise NotImplementedError("Async not implemented.")
