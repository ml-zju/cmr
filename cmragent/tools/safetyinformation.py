from langchain.llms import BaseLLM
from .base_pubchem_tool import BasePubchemTool

def create_safety_tool(property_name: str, property_description: str):
    class SafetyTool(BasePubchemTool):
        name: str = property_name.lower()
        description: str = f"""Tool to find the {property_name.lower()} information of chemical compounds. 
        Input: Chemical name, CAS number, or SMILES notation
        Output: {property_name} information as reported by PubChem"""

        def __init__(self, llm: BaseLLM):
            prompt_template = f"Extract the {property_name.lower()} information from: {{data}}"
            super().__init__(llm, [property_name], prompt_template)

    return SafetyTool

SAFETY_PROPERTY_TOOLS = {
    "AcceptableDailyIntakes": ("Acceptable Daily Intakes", "Extracts data about the Acceptable Daily Intakes (ADI) of the chemical by humans or animals."),
    "FireHazards": ("Fire Hazards", "Extracts information about the chemical's fire hazards, flammability characteristics, and potential fire-related risks."),
    "ExplosionHazards": ("Explosion Hazards", "Extracts data about explosion risks, detonation conditions, and explosive properties of the chemical."),
    "FirePotential": ("Fire Potential", "Provides information about the chemical's potential to ignite, sustain combustion, and contribute to fire hazards."),
    "FirstAid": ("First Aid", "Extracts first aid procedures and immediate medical response requirements for exposure to the chemical."),
    "FireFightingProcedures": ("Fire Fighting Procedures", "Details specific procedures, equipment, and methods required for fighting fires involving this chemical."),
    "PreventiveMeasures": ("Preventive Measures", "Extracts information about preventive measures and safety precautions for handling the chemical."),
    "SpillageDisposal": ("Spillage Disposal", "Extracts procedures and methods for safely disposing of chemical spills."),
    "CleanupMethods": ("Cleanup Methods", "Details proper methods and procedures for cleaning up chemical spills and contamination."),
    "DisposalMethods": ("Disposal Methods", "Provides information about proper disposal methods and requirements for the chemical."),
    "ShipmentMethodsRegulations": ("Shipment Methods and Regulations", "Details regulations and approved methods for shipping and transporting the chemical."),
    "AllowableTolerances": ("Allowable Tolerances", "Specifies the maximum amount of chemical residues (e.g., pesticides) allowed to remain in or on food products such as fruits, vegetables, and grains. Also known as maximum residue limits (MRLs) in some countries."),
    "ExplosiveLimitsAndPotential": ("Explosive Limits and Potential", "Extracts data on explosive limits (e.g., LEL, UEL) and the overall explosive potential of the chemical."),
    "FirefightingHazards": ("Firefighting Hazards", "Describes specific hazards that may be encountered by personnel during firefighting activities involving the chemical."),
    "FlammableLimits": ("Flammable Limits", "Extracts data on the flammable limits (e.g., LFL/LEL, UFL/UEL) of the chemical in air."),
    "ImmediatelyDangerousToLifeOrHealth": ("Immediately Dangerous to Life or Health", "Extracts information on Immediately Dangerous to Life or Health (IDLH) concentrations and conditions for the chemical."),
    "PackagingAndLabelling": ("Packaging and Labelling", "Details safety considerations, requirements, and best practices for packaging and labelling the chemical."),
    "StabilityShelfLife": ("Stability/Shelf Life", "Extracts information regarding the chemical's stability, shelf life, and decomposition hazards from a safety perspective."),
    "ToxicCombustionProducts": ("Toxic Combustion Products", "Extracts information on toxic gases or particulates produced during the combustion of the chemical."),
    "ImmediatelyDangerousToLifeOrHealth": ("Immediately Dangerous to Life or Health (IDLH)", "Provides IDLH concentrations and conditions."),
    "PackagingAndLabelling": ("Packaging and Labelling", "Details safety requirements for packaging and labelling."),
    "PPE": ("Personal Protective Equipment (PPE)", "Recommends personal protective equipment for handling."),
    "ToxicCombustionProducts": ("Toxic Combustion Products", "Lists toxic gases or particulates from combustion."),
    "EffectsLongTermExposure": ("Effects of Long Term Exposure", "Health effects from chronic (long-term) exposure."),
    "EffectsShortTermExposure": ("Effects of Short Term Exposure", "Health effects from acute (short-term) exposure."),
    "IsolationAndEvacuation": ("Isolation and Evacuation", "Measures for isolation and evacuation in emergencies."),
    "InhalationRisk": ("Inhalation Risk", "Assesses risk from inhalation exposure."),
    "OEL": ("Occupational Exposure Limits (OEL)", "Occupational exposure limit for workplace safety."),
    "PEL": ("Permissible Exposure Limit (PEL)", "Permissible exposure limit under OSHA."),
    "REL": ("Recommended Exposure Limit (REL)", "Recommended exposure limit for workplace safety."),
    "SkinEyeRespiratoryIrritations": ("Skin, Eye, and Respiratory Irritations", "Describes irritations caused by exposure."),
    "TLV": ("Threshold Limit Values (TLV)", "Threshold limit value for safe long-term exposure."),
}


for property_name, (name, description) in SAFETY_PROPERTY_TOOLS.items():
    globals()[f"{name}Tool"] = create_safety_tool(name, description)