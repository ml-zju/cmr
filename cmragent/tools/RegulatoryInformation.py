from langchain.llms import BaseLLM
from .base_pubchem_tool import BasePubchemTool

def create_regulatory_tool(property_name: str, property_description: str):
    class RegulatoryTool(BasePubchemTool):
        name: str = property_name.lower()
        description: str = f"""Tool to extract regulatory information about the chemical compound, such as its environmental, health, or safety-related regulations.
        Input: Chemical name, CAS number, or SMILES notation
        Output: {property_name} regulatory information as reported by regulatory agencies."""

        def __init__(self, llm: BaseLLM):
            prompt_template = f"Extract the {property_name.lower()} information from: {{data}}"
            super().__init__(llm, [property_name], prompt_template)

    return RegulatoryTool

REGULATORY_PROPERTY_TOOLS = {
    "AtmosphericStandards": ("Atmospheric Standards", "This section provides atmospheric standards related to the chemical, established by regulatory agencies."),
    "CERCLAReportableQuantities": ("CERCLA Reportable Quantities", "The CERCLA reportable quantity for a hazardous substance is the minimum quantity of the substance which, if released, must be reported, under the Comprehensive Environmental Response, Compensation, and Liability Act of 1980 (CERCLA, also commonly known as Superfund)."),
    "CleanWaterActRequirements": ("Clean Water Act Requirements", "The Clean Water Act (CWA) of 1972 establishes the basic structure for regulating discharges of pollutants into the waters of the United States and regulating quality standards for surface waters. Under CWA, the U.S. Environmental Protection Agency (EPA) developed the Toxic Pollutant List and the Priority Pollutant List."),
    "DHSCOI": ("DHS Chemicals of Interest (COI)", "This section provides the Department of Homeland Security (DHS) Chemicals of Interest (COI) and related information."),
    "FDARequirements": ("FDA Requirements", "FDA requirements regarding this chemical and products containing it. FDA Requirements means any requirements of the Federal Food, Drug and Cosmetic Act (FDCA), as amended, and any rules or regulations promulgated thereunder."),
    "FederalDrinkingWaterGuidelines": ("Federal Drinking Water Guidelines", "Federal drinking water guidelines (e.g. maximum containment level (MCL)) for this chemical. These guidelines are recommendations and not legally enforceable."),
    "FederalDrinkingWaterStandards": ("Federal Drinking Water Standards", "Federal drinking water standards (e.g. maximum containment level (MCL)) for this chemical. These standards are legally enforceable."),
    "FIFRARequirements": ("FIFRA Requirements", "The Federal Insecticide, Fungicide, and Rodenticide Act (FIFRA) is the Federal statute that governs the registration, distribution, sale, and use of pesticides in the United States."),
    "RCRARequirements": ("RCRA Requirements", "The Resource Conservation and Recovery Act (RCRA) is the public law that creates the framework for the proper management of hazardous and non-hazardous solid waste."),
    "StateDrinkingWaterGuidelines": ("State Drinking Water Guidelines", "State drinking water guidelines (e.g. maximum containment level (MCL)) for this chemical. In general, these guidelines are recommendations and not legally enforceable."),
    "StateDrinkingWaterStandards": ("State Drinking Water Standards", "State drinking water standards (e.g. maximum containment level (MCL)) for this chemical. These standards are legally enforceable."),
    "TSCARequirements": ("TSCA Requirements", "The Toxic Substances Control Act (TSCA) of 1976 provides the U.S. Environmental Protection Agency (EPA) authority to regulate the introduction of new or already existing chemicals."),
}

for property_name, (name, description) in REGULATORY_PROPERTY_TOOLS.items():
    globals()[f"{name}Tool"] = create_regulatory_tool(name, description)

