import pandas as pd
from typing import Dict
from agents.agent import CMR
from tools import load_llm_tools
from llm.utils import get_llm
from rdkit import Chem
from tools.PhysicalProperties import PROPERTY_TOOLS, create_property_tool
from tools.healthInformation import HEALTH_EFFECT_TOOLS, create_health_tool
from tools.EnvironmentalInformation import ENVIRONMENTAL_PROPERTY_TOOLS, create_environment_tool

PROPERTY_CATEGORIES = {
    "physical": {
        "title": "Physical Properties",
        "properties": [
            "PhysicalDescription", "Density", "BoilingPoint", "MeltingPoint", "VaporPressure", "LogP", "Solubility",
            "StabilityShelfLife",
            "Odor", "Taste", "ColorForm", "Solubility",
        ]
    },
    "environmental": {
        "title": "Environmental Properties",
        "properties": [
            "EnvironmentalBioconcentration", "EnvironmentalBiodegradation", "EnvironmentalFate", "EcotoxicityValues",
        ]
    },
    "cmr": {
        "title": "CMR Properties",
        "properties": [
            "Carcinogenic", "Mutagenic", "Toxic to Reproduction"
        ]
    },
    "health": {
        "title": "Health Properties",
        "properties": [
            "ExposureRoutes", "AdverseEffects", "ToxicologicalInformation", "EvidenceForCarcinogenicity",
            "HealthEffects", "ToxicitySummary", "ToxicityData",
            "HumanToxicityExcerpts",
        ]
    }
}


def validate_smiles(smiles: str) -> tuple[bool, str]:
    if not smiles:
        return False, "No SMILES string provided."
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False, "Invalid SMILES string."
    return True, ""


def load_property_tools(llm) -> Dict:
    property_tools = {}
    for tool_name, (display_name, description) in PROPERTY_TOOLS.items():
        ToolClass = create_property_tool(display_name, description)
        property_tools[tool_name] = ToolClass(llm)
    return property_tools


def load_environmental_tools(llm) -> Dict:
    environmental_tools = {}
    for tool_name, (display_name, description) in ENVIRONMENTAL_PROPERTY_TOOLS.items():
        ToolClass = create_environment_tool(display_name, description)
        environmental_tools[tool_name] = ToolClass(llm)
    return environmental_tools


def load_health_tools(llm) -> Dict:
    health_tools = {}
    for tool_name, (display_name, description) in HEALTH_EFFECT_TOOLS.items():
        ToolClass = create_health_tool(display_name, description)
        health_tools[tool_name] = ToolClass(llm)
    return health_tools


def load_all_tools(llm) -> Dict:
    tools = {
        "physical": load_property_tools(llm),
        "environmental": load_environmental_tools(llm),
        "health": load_health_tools(llm)
    }
    return tools


def get_summary_prompt(category, category_data, info):
    base_data = f"{chr(10).join(category_data)}\n\n"
    prompts = {
        "physical": (
            f"Based on the following physical properties:\n{base_data}"
            "Provide a concise summary focusing on the key physical characteristics. "
            "Explain what these properties suggest about the compound's behavior under normal conditions. "
            "Highlight any notable or unusual physical properties. "
            "Summarize in 3-5 sentences."
        ),
        "environmental": (
            f"Based on the following environmental data:\n{base_data}"
            "Analyze the environmental impact and behavior of this compound. "
            "Consider bioconcentration, biodegradation, and ecotoxicity aspects. "
            "Evaluate the potential environmental risks and persistence. "
            "Summarize in 3-5 sentences."
        ),
        "cmr": (
            f"Based on the following CMR predictions:\n{base_data}"
            "Prediction the carcinogenic, mutagenic, and reproductive toxicity risks. "
            "Don't Explain the significance of these predictions for human safety. "
            "Identify which aspects pose the most significant concerns. "
            "Summarize in 3-5 sentences."
        ),
        "health": (
            f"Based on the following toxicological and health data:\n{base_data}"
            "Explain the result of cmr prediction (whether the prediction is true). "
            "Analyze the primary health concerns and toxic effects. "
            "Consider both acute and chronic exposure scenarios. "
            "Highlight the most critical health hazards and exposure routes. "
            "Discuss the overall toxicological profile. "
            "Summarize in 5-8 sentences."
        )
    }
    return prompts.get(category, "")


class CMRReportTool:

    def __init__(self, llm, **kwargs):
        self.llm = llm
        self.tools = load_llm_tools(self.llm, verbose=False)
        self.model = CMR(llm=self.llm, tools=self.tools)
        self.all_tools = load_all_tools(self.llm)

    def predict(self, smiles: str) -> Dict:
        is_valid, error_message = validate_smiles(smiles)
        if not is_valid:
            raise ValueError(error_message)

        results = {}
        molecule_info_tools = ["CASRetriever", "NameRetriever", "FormulaRetriever"]
        for tool_name in molecule_info_tools:
            for tool in self.model.tools:
                if tool.name == tool_name:
                    results[tool_name] = tool.run(smiles)

        for category, info in PROPERTY_CATEGORIES.items():
            category_data = []
            category_tools = self.all_tools.get(category, {})

            for property_name in info["properties"]:
                try:
                    if category == "cmr":
                        tool_input = str({
                            'smiles': [smiles],
                            'property_type': property_name
                        })
                        for tool in self.model.tools:
                            if tool.name == "CMR Prediction":
                                results_df = tool.run(tool_input)
                                if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                                    results[f"{property_name}_Prediction"] = {
                                        'probability_0': float(results_df.iloc[0]['probability_class_0']),
                                        'probability_1': float(results_df.iloc[0]['probability_class_1']),
                                        'prediction': 'Positive' if results_df.iloc[0][
                                                                        'prediction'] == 1 else 'Negative'
                                    }
                                    category_data.append(
                                        f"{property_name}: {results[f'{property_name}_Prediction']['prediction']}")
                                break
                    elif property_name in category_tools:
                        tool = category_tools[property_name]
                        result = tool.run(smiles)
                        results[property_name] = result
                        category_data.append(f"{property_name}: {result}")
                except Exception as e:
                    results[property_name] = "Not Available"
                    category_data.append(f"{property_name}: Not Available")

            if category_data:
                summary_prompt = get_summary_prompt(category, category_data, info)
                try:
                    category_summary = self.model.llm.predict(summary_prompt)
                    results[f"{category}_summary"] = category_summary
                except Exception:
                    results[f"{category}_summary"] = "Summary generation failed"

        return results

    def generate_report(self, results: Dict, smiles: str) -> str:
        section_mapping = {
            "physical": "II. Physicochemical Characterization",
            "environmental": "III. Environmental Fate and Behavior Analysis",
            "cmr": "IV. Carcinogenic, Mutagenic, and Reproductive Toxicity Prediction",
            "health": "V. Toxicological Profile and Health Hazard Evaluation"
        }
        analysis_titles = {
            "physical": "Physicochemical Properties Analysis",
            "environmental": "Environmental Impact Assessment",
            "cmr": "CMR Hazard Evaluation",
            "health": "Health Risk Assessment"
        }
        report_content = []
        report_content.append("# Prediction Report of CMR")
        report_content.append("*A Systematic Analysis of Physicochemical, Toxicological, and Environmental Properties*")
        report_content.append("\n---\n")
        report_content.append("## I. Molecular Characterization")
        report_content.append(f"Systematic Nomenclature: {results.get('NameRetriever', 'Not Available')}")
        report_content.append(f"CAS Number: {results.get('CASRetriever', 'Not Available')}")
        report_content.append(f"Molecular Formula: {results.get('FormulaRetriever', 'Not Available')}")
        report_content.append(f"SMILES Notation: {smiles}")

        for category, info in PROPERTY_CATEGORIES.items():
            report_content.append(f"\n## {section_mapping[category]}")
            report_content.append("\n### Analytical Parameters and Results")
            for prop in info["properties"]:
                if category == "cmr":
                    prediction_key = f"{prop}_Prediction"
                    result = results.get(prediction_key, {})
                    if isinstance(result, dict):
                        prob_0 = result.get('probability_0', 'N/A')
                        prob_1 = result.get('probability_1', 'N/A')
                        prediction = result.get('prediction', 'Not Available')
                        report_content.append(f"{prop}:")
                        prob_0_str = f"- Probability (Negative): {float(prob_0):.3f}" if isinstance(prob_0, (
                        float, int)) else f"- Probability (Negative): {prob_0}"
                        prob_1_str = f"- Probability (Positive): {float(prob_1):.3f}" if isinstance(prob_1, (
                        float, int)) else f"- Probability (Positive): {prob_1}"
                        report_content.append(prob_0_str)
                        report_content.append(prob_1_str)
                        report_content.append(f"- Prediction: {prediction}")
                else:
                    value = results.get(prop, "Not Available")
                    report_content.append(f"{prop}: {value}")

            if f"{category}_summary" in results:
                report_content.append(f"\n### {analysis_titles[category]}")
                report_content.append(results[f"{category}_summary"])

        report_content.append("\n## VII. Methodology and Limitations")
        report_content.append("""
            ### Technical Notes and Methodological Considerations

            **Analytical Methodology:**
            - Computational predictions based on validated QSAR models
            - Structure-activity relationship analysis
            - Integration of multiple prediction algorithms

            **Data Quality Considerations:**
            - Predictions are based on validated computational models
            - Results should be interpreted within the applicability domain
            - Experimental validation is recommended for critical parameters

            **Uncertainty Assessment:**
            - Predictions include inherent uncertainty levels
            - Results should be considered as screening-level assessments
            - Further testing may be required for regulatory purposes
                """)
        report_content.append("\n---")
        report_content.append(f"""
Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Protocol Version: 2.0
Computational Assessment Framework: CMR Prediction Suite
        """)
        return "\n".join(report_content)

    def run(self, smiles: str) -> Dict:
        return self.predict(smiles)

    def report(self, smiles: str) -> str:
        results = self.predict(smiles)
        return self.generate_report(results, smiles)