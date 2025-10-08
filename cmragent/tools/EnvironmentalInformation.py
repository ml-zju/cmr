from langchain.llms import BaseLLM
from .base_pubchem_tool import BasePubchemTool

def create_environment_tool(property_name: str, property_description: str):
    class EnvironmentalTool(BasePubchemTool):
        name: str = property_name.lower()
        description: str = f"""Tool to find the {property_name.lower()} of chemical compounds.
        Input: Chemical name, CAS number, or SMILES notation
        Output: {property_name} value as reported by PubChem"""

        def __init__(self, llm: BaseLLM):
            prompt_template = f"Extract the {property_name.lower()} value from: {{data}}"
            super().__init__(llm, [property_name], prompt_template)

    return EnvironmentalTool

ENVIRONMENTAL_PROPERTY_TOOLS = {
    "AnimalConcentrations": ("Animal Concentrations", "Extracts the concentrations of the chemical in various animal species after exposure."),
    "ArtificialPollutionSources": ("Artificial Pollution Sources", "Extracts information about artificial pollution sources of the chemical."),
    "AtmosphericConcentrations": ("Atmospheric Concentrations", "Extracts information about the concentration of the chemical in the atmosphere."),
    "AverageDailyIntakeTool": ("Average Daily Intake", "Describes the average amount of the compound taken into the body through eating, drinking, or breathing."),
    "BodyBurden": ("Body Burden", "Extracts information about the amount of chemical stored or accumulated in the body."),
    "EcotoxicityExcerpts": ("Ecotoxicity Excerpts", "Extracts excerpts about the chemical's ecotoxicity."),
    "EcotoxicityValues": ("Ecotoxicity Values", "Extracts numeric ecotoxicity values such as LC50, EC50, NOEC."),
    "EffluentConcentrations": ("Effluent Concentrations", "Extracts data on the concentration of the chemical in industrial or municipal effluents."),
    "EnvironmentalAbioticDegradation": ("Environmental Abiotic Degradation", "Extracts information about non-biological degradation of the chemical."),
    "EnvironmentalBioconcentration": ("Environmental Bioconcentration", "Extracts information on the bioconcentration of the chemical in organisms."),
    "EnvironmentalBiodegradation": ("Environmental Biodegradation", "Extracts data on the microbial breakdown of the chemical."),
    "EnvironmentalFate": ("Environmental Fate", "Extracts information about the distribution, transformation, and fate of the chemical in the environment."),
    "EnvironmentalFateExposureSummary": ("Environmental Fate/Exposure Summary", "Provides a summary of the chemical's environmental fate and exposure."),
    "EnvironmentalWaterConcentrations": ("Environmental Water Concentrations", "Extracts data about the concentration of the chemical in different water bodies."),
    "EPAEcotoxicity": ("EPA Ecotoxicity", "Extracts ecotoxicity data from the U.S. Environmental Protection Agency (EPA)."),
    "FishSeafoodConcentrations": ("Fish / Seafood Concentrations", "Extracts information about the concentration of the chemical in fish and seafood."),
    "FoodSurveyValues": ("Food Survey Values", "Extracts data from food surveys regarding chemical concentrations."),
    "ICSCEnvironmentalData": ("ICSC Environmental Data", "Extracts environmental data from International Chemical Safety Cards."),
    "MilkConcentrations": ("Milk Concentrations", "Extracts data about the concentration of the chemical in milk."),
    "NaturalPollutionSources": ("Natural Pollution Sources", "Extracts information about natural sources of pollution from the chemical."),
    "OtherEnvironmentalConcentrations": ("Other Environmental Concentrations", "Extracts concentration data from other environmental sources not covered elsewhere."),
    "PlantConcentrations": ("Plant Concentrations", "Extracts information about the concentration of the chemical in plants."),
    "ProbableRoutesHumanExposure": ("Probable Routes of Human Exposure", "Describes the probable routes through which humans may be exposed to the chemical."),
    "SedimentSoilConcentrations": ("Sediment / Soil Concentrations", "Extracts data about the concentration of the chemical in sediments and soils."),
    "SoilAdsorptionMobility": ("Soil Adsorption/Mobility", "Extracts information on the chemical's soil adsorption and mobility."),
    "VolatilizationFromWaterSoil": ("Volatilization from Water / Soil", "Extracts data on the chemical's volatilization potential."),
    "AirWaterReactions": ("Air and Water Reactions", "Extracts data on the chemical's reactivity or transformation in air and water environments."),
    "EffectsShortTermExposure": ("Effects of Short Term Exposure", "Describes the health effects observed after brief or acute exposure to the chemical."),
}

# Dynamically create all environmental property tools
for property_name, (name, description) in ENVIRONMENTAL_PROPERTY_TOOLS.items():
    globals()[f"{name}Tool"] = create_environment_tool(name, description)
