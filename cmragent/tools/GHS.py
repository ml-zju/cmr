from langchain.tools import BaseTool
import requests
from pydantic import Field
from typing import Dict, Any, List, Optional
from .get_cid import get_compound_cid

class GHSClassificationTool(BaseTool):
    name: str = "GHSClassification"
    description: str = (
        "Fetches the GHS (Globally Harmonized System of Classification and Labelling of Chemicals) classification "
        "and safety data for a given chemical compound using its CAS number. Input must be a valid CAS number."
    )
    pubchem_data: Dict[str, Any] = Field(default_factory=dict)

    def _fetch_pubchem_data(self, identifier: str) -> Dict[str, Any]:
        if identifier not in self.pubchem_data:
            try:
                cid = get_compound_cid(identifier)
                url2 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
                compound_response = requests.get(url2, timeout=10)
                compound_response.raise_for_status()
                self.pubchem_data[identifier] = compound_response.json()
            except requests.exceptions.RequestException as e:
                return {"error": f"Network error while fetching PubChem data: {str(e)}"}
            except Exception as e:
                return {"error": f"Error fetching PubChem data: {str(e)}"}
        return self.pubchem_data[identifier]

    def _ghs_classification(self, cas_number: str) -> str:
        """
        Extracts GHS classification from PubChem data.
        """
        data = self._fetch_pubchem_data(cas_number)
        if isinstance(data, dict) and "error" in data:
            return data["error"]

        try:
            for section in data["Record"]["Section"]:
                if section.get("TOCHeading") == "Chemical Safety":
                    ghs = [
                        markup["Extra"]
                        for markup in section["Information"][0]["Value"]["StringWithMarkup"][0]["Markup"]
                    ]
                    if ghs:
                        return ", ".join(ghs)
        except (KeyError, IndexError):
            pass
        return "No GHS classification found in PubChem."

    @staticmethod
    def _scrape_pubchem(data: Dict[str, Any], heading1: str, heading2: str, heading3: str) -> Optional[List[Dict[str, Any]]]:
        try:
            filtered_sections = []
            for section in data["Record"]["Section"]:
                if section.get("TOCHeading") == heading1:
                    for section2 in section.get("Section", []):
                        if section2.get("TOCHeading") == heading2:
                            for section3 in section2.get("Section", []):
                                if section3.get("TOCHeading") == heading3:
                                    filtered_sections.append(section3)
            return filtered_sections
        except Exception:
            return None

    def _get_safety_data(self, cas: str) -> List[Optional[Dict[str, Any]]]:
        data = self._fetch_pubchem_data(cas)
        if isinstance(data, dict) and "error" in data:
            return [{"error": data["error"]}]

        safety_data = []
        iterations = [
            (
                [
                    "Health Hazards",
                    "GHS Classification",
                    "NFPA Hazard Classification",
                    "Hazard Classes and Categories",
                    "Highly Hazardous Substance"
                ],
                "Safety and Hazards",
                "Hazards Identification",
            ),
        ]

        for items, header1, header2 in iterations:
            for item in items:
                result = self._scrape_pubchem(data, header1, header2, item)
                if result:
                    safety_data.append({item: result})
                else:
                    safety_data.append({item: "No data found"})

        return safety_data

    def _run(self, cas_number: str) -> Dict[str, Any]:
        ghs_classification = self._ghs_classification(cas_number)
        safety_data = self._get_safety_data(cas_number)
        return {
            "GHS Classification": ghs_classification,
            "Safety Data": safety_data
        }

    async def _arun(self, cas_number: str) -> Dict[str, Any]:
        return self._run(cas_number)

