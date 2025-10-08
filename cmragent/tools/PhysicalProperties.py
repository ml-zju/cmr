from langchain.llms import BaseLLM
from .base_pubchem_tool import BasePubchemTool

def create_property_tool(property_name: str, property_description: str):
    class PropertyTool(BasePubchemTool):
        name: str = property_name.lower()
        description: str = f"""Tool to find the {property_name.lower()} of chemical compounds. 
        Input: Chemical name, CAS number, or SMILES notation
        Output: {property_name} value as reported by PubChem"""

        def __init__(self, llm: BaseLLM):
            prompt_template = f"Extract the {property_name.lower()} value from: {{data}}"
            super().__init__(llm, [property_name], prompt_template)

    return PropertyTool


PROPERTY_TOOLS = {
    "PhysicalDescription": ("Physical Description", "Extracts the physical description from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "Density": ("Density", "Extracts the Density from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "BoilingPoint": ("Boiling Point", "Extracts the Boiling Point from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "MeltingPoint": ("Melting Point", "Extracts the Melting Point from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "HeatOfCombustion": ("Heat Of Combustion", "Extracts the Heat of Combustion from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "HeatOfVaporization": ("Heat Of Vaporization", "Extracts the Heat of Vaporization from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "EnthalpyOfSublimation": ("Enthalpy Of Sublimation", "Extracts the Enthalpy of Sublimation from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "Decomposition": ("Decomposition", "Extracts the Decomposition property from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "AutoignitionTemperature": ("Auto ignition Temperature", "Extracts the Auto ignition Temperature from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "VaporPressure": ("Vapor Pressure", "Extracts the Vapor Pressure from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "VaporDensity": ("Vapor Density", "Extracts the Vapor Density from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "SurfaceTension": ("Surface Tension", "Extracts the Surface Tension from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "Viscosity": ("Viscosity", "Extracts the Viscosity from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "ColorForm": ("Color/Form", "Extracts the Color/Form from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "Odor": ("Odor", "Extracts the Odor from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "Taste": ("Taste", "Extracts the Taste from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "FlashPoint": ("Flash Point", "Extracts the Flash Point from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "StabilityShelfLife": ("Stability / ShelfLife", "Extracts the Stability/Shelf Life from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "Corrosivity": ("Corrosivity", "Extracts the Corrosivity from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "Polymerization": ("Polymerization", "Extracts the Polymerization from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "SADT": ("SelfAcceleratingDecompositionTemperature", "Extracts the Self-Accelerating Decomposition Temperature (SADT) from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "Solubility": ("Solubility", "Extracts the Solubility from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "DissociationConstants": ("Dissociation Constants", "Extracts the Dissociation Constants from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "pH": ("pH", "Extracts the pH from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "LogP": ("LogP", "Extracts the LogP from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "LogS": ("LogS", "Extracts the LogS from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "HenrysLawConstant": ("Henrys Law Constant", "Extracts Henry's Law Constant from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "RefractiveIndex": ("Refractive Index", "Extracts the Refractive Index from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "OpticalRotation": ("Optical Rotation", "Extracts the Optical Rotation from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "DielectricConstant": ("Dielectric Constant", "Extracts the Dielectric Constant from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "IonizationPotential": ("Ionization Potential", "Extracts the Ionization Potential from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "IonizationEfficiency": ("Ionization Efficiency", "Extracts the Ionization Efficiency from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "IsoelectricPoint": ("IsoelectricPoint", "Extracts the Isoelectric Point from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "ARC": ("Accelerating Rate Calorimetry", "Extracts the Accelerating Rate Calorimetry (ARC) from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "CollisionCrossSection": ("Collision Cross Section", "Extracts the Collision Cross Section from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "Caco2Permeability": ("Caco2 Permeability", "Extracts the Caco2 Permeability from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "KovatsRetentionIndex": ("Kovats Retention Index", "Extracts the Kovats Retention Index from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "RelativeEvaporationRate": ("Relative Evaporation Rate", "Extracts the Relative Evaporation Rate from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "Dispersion": ("Dispersion", "Extracts the Dispersion from the provided input. Please input with the name or CAS/SMILES of molecule."),
    "OtherExperimentalProperties": ("Other Experimental Properties", "Extracts the Other Experimental Properties from the provided input. Please input with the name or CAS/SMILES of molecule."),
}

for property_name, (name, description) in PROPERTY_TOOLS.items():
    globals()[f"{name}Property"] = create_property_tool(name, description)



