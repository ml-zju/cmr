from langchain.llms import BaseLLM
from .base_pubchem_tool import BasePubchemTool

def create_health_tool(property_name: str, property_description: str):
    class HealthEffectTool(BasePubchemTool):
        name: str = property_name.lower()
        description: str = f"""Tool to find the {property_name.lower()} of chemical compounds. 
        Input: Chemical name, CAS number, or SMILES notation
        Output: {property_name} value as reported by PubChem"""

        def __init__(self, llm: BaseLLM):
            prompt_template = f"Extract the {property_name.lower()} value from: {{data}}"
            super().__init__(llm, [property_name], prompt_template)

    return HealthEffectTool

HEALTH_EFFECT_TOOLS = {
    "AntidoteAndEmergencyTreatment": ("Antidote and Emergency Treatment",
                                     "Provides information on potential antidotes and recommended emergency treatment protocols for exposure to the chemical. This may include steps for handling poisoning or exposure and specific medications or treatments used in such emergencies."),
    "DrugInducedLiverInjury": ("Drug Induced Liver Injury",
                               "Extracts data on the potential for the chemical to cause liver damage, based on the FDA’s Drug-Induced Liver Injury Rank (DILIrank) database. This includes categorizing the substance’s risk as high, moderate, low, or uncertain and describing the severity of liver injury."),
    "EffectsDuringPregnancyAndLactation": ("Effects During Pregnancy and Lactation",
                                          "Provides information on any potential adverse effects the chemical may have during pregnancy or lactation. This includes the risks to the fetus or infant when the mother is exposed."),
    "EvidenceForCarcinogenicity": ("Evidence for Carcinogenicity",
                                  "Extracts the evidence and studies that support or refute the carcinogenic potential of the chemical, drawing from various sources including the Hazardous Substances Data Bank (HSDB) and other scientific publications. Input should include the molecule's name or CAS/SMILES."),
    "HealthEffects": ("Health Effects",
                     "Summarizes the principal health effects that may result from exposure to the chemical. This includes potential organ toxicity, neurological impacts, or other systemic health concerns, as recorded by various health and safety databases."),
    "Hepatotoxicity": ("Hepatotoxicity",
                      "Provides detailed information on the chemical’s hepatotoxicity, including the effects on liver function, the frequency of liver enzyme elevations, and the character of liver injury observed in clinical settings. Input should include the molecule’s name or CAS/SMILES."),
    "Interactions": ("Interactions",
                    "Provides information on any known chemical interactions between the molecule and other substances. This may include synergistic or antagonistic effects, and is useful for understanding the broader context of safety and toxicity."),
    "MedicalSurveillance": ("Medical Surveillance",
                           "Describes the recommended medical surveillance protocols for individuals exposed or potentially exposed to the chemical. This typically includes routine testing, monitoring for adverse effects, and steps to ensure early detection of health issues."),
    "PopulationsAtSpecialRisk": ("Populations at Special Risk",
                                 "Identifies specific groups of people or organisms that may be particularly vulnerable to exposure to the chemical, such as pregnant women, children, or workers in high-risk environments."),
    "TargetOrgans": ("Target Organs",
                     "Identifies the organs that are most affected by exposure to the chemical. This can include organs like the liver, lungs, kidneys, or central nervous system, based on existing medical or toxicological data."),
    "CancerSites": ("Cancer Sites", "Extracts information on specific organs or tissues where the chemical has been shown to cause cancer."),
    "CarcinogenClassification": ("Carcinogen Classification", "Provides the chemical's carcinogen classification from various agencies (e.g., IARC, NTP, EPA)."),
    "EPAHumanHealthBenchmarksForPesticides": ("EPA Human Health Benchmarks for Pesticides", "Extracts Human Health Benchmarks for Pesticides from the U.S. Environmental Protection Agency (EPA)."),
    "EPAProvisionalPeerReviewedToxicityValues": ("EPA Provisional Peer-Reviewed Toxicity Values", "Extracts Provisional Peer-Reviewed Toxicity Values (PPRTVs) from the U.S. EPA."),
    "ExposureRoutes": ("Exposure Routes", "Describes the primary routes (e.g., inhalation, ingestion, dermal) by which humans can be exposed to the chemical and associated health implications. Input should be the molecule's name or CAS/SMILES."),
    "MinimumRiskLevel": ("Minimum Risk Level", "Extracts Minimum Risk Levels (MRLs) for the chemical, which are estimates of daily human exposure that are unlikely to cause appreciable non-cancer health effects over a specified duration of exposure. Input should be the molecule's name or CAS/SMILES."),
    "NIOSHToxicityData": ("NIOSH Toxicity Data", "Extracts toxicity data and occupational health information from the National Institute for Occupational Safety and Health (NIOSH). Input should be the molecule's name or CAS/SMILES."),
    "ProteinBinding": ("Protein Binding", "Extracts data on the extent to which the chemical binds to plasma proteins, which can affect its distribution and availability. Input should be the molecule's name or CAS/SMILES."),
    "RAISToxicityValues": ("RAIS Toxicity Values", "Extracts toxicity values and related data from the Risk Assessment Information System (RAIS). Input should be the molecule's name or CAS/SMILES."),
    "Treatment": ("Treatment", "Provides information on medical treatments for poisoning or adverse health effects resulting from exposure to the chemical, potentially including supportive care and specific therapies. Input should be the molecule's name or CAS/SMILES."),
    "TSCATestSubmissions": ("TSCA Test Submissions", "Extracts data from test submissions made under the Toxic Substances Control Act (TSCA). Input should be the molecule's name or CAS/SMILES."),
    "USGSWaterQualityScreening": ("USGS Health-Based Screening Levels for Evaluating Water-Quality", "Extracts Health-Based Screening Levels (HBSLs) from the U.S. Geological Survey (USGS) for evaluating water-quality data. Input should be the molecule's name or CAS/SMILES.")
}


# Dynamically create all health effect tools
for property_name, (name, description) in HEALTH_EFFECT_TOOLS.items():
    globals()[f"{name}Tool"] = create_health_tool(name, description)

