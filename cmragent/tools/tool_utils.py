from typing import List, Type
from langchain.base_language import BaseLanguageModel
from langchain.tools import Tool, BaseTool
from langchain.agents import load_tools as _load_external_tools
from tools.GHS import GHSClassificationTool
from tools.molecular_transfer import CASRetriever, SMILESRetriever, NameRetriever, InChIRetriever, InChIKeyRetriever, FormulaRetriever
from tools.EnvironmentalInformation import ENVIRONMENTAL_PROPERTY_TOOLS, create_environment_tool
from tools.healthInformation import HEALTH_EFFECT_TOOLS, create_health_tool
from tools.PhysicalProperties import PROPERTY_TOOLS, create_property_tool
from tools.safetyinformation import SAFETY_PROPERTY_TOOLS, create_safety_tool
from tools.RegulatoryInformation import REGULATORY_PROPERTY_TOOLS, create_regulatory_tool

from tools.AcuteToxicity import AcuteToxicityTool
from tools.Symptoms import SymptomsTool
from tools.HumanToxicity import HumanToxicityTool
from tools.exposureconcentrations import EnvironmentalConcentrationsTool
from tools.NonHumanToxicity import NonHumanToxicityTool
from tools.SafeStorage import SafeStorageTool
from tools.HazardsSummary import HazardsSummaryTool
from tools.EmergencyResponseMeasures import EmergencyResponseMeasuresTool

from tools.cmr_screen import CMRPrediction
from tools.CarcinogenicityTool import ComprehensiveCarcinogenicityTool
from tools.MutagenicityTool import ComprehensiveMutagenicityTool
from tools.ReproductiveToxicityTool import ComprehensiveReproductiveToxicityTool

_EXTERNAL_TOOL_NAMES = ['llm-math']

def load_llm_tools(
        llm: BaseLanguageModel,
        verbose: bool = False,
        search_internet: bool = False
) -> List[BaseTool]:
    if verbose:
        print("🔧 Loading LLM tools...")

    loaded_tools: List[BaseTool] = []
    seen_names = set()

    ext_tools = _load_external_tools(_EXTERNAL_TOOL_NAMES, llm=llm)
    for tool in ext_tools:
        if tool.name not in seen_names:
            loaded_tools.append(tool)
            seen_names.add(tool.name)

    tool_sets = [
        (PROPERTY_TOOLS, create_property_tool, "{}Tool"),
        (HEALTH_EFFECT_TOOLS, create_health_tool, "{}Tool"),
        (ENVIRONMENTAL_PROPERTY_TOOLS, create_environment_tool, "{}Tool"),
        (SAFETY_PROPERTY_TOOLS, create_safety_tool, "{}Tool"),
        (REGULATORY_PROPERTY_TOOLS, create_regulatory_tool, "{}Tool"),
    ]

    for tool_dict, factory_fn, name_fmt in tool_sets:
        for key, (display_name, description) in tool_dict.items():
            class_name = display_name.replace(" ", "")
            tool_name = name_fmt.format(display_name.replace(" ", ""))
            if tool_name in seen_names:
                continue

            tool_cls: Type = globals().get(f"{class_name}{name_fmt.strip('{}')}")
            if not tool_cls:
                tool_cls = factory_fn(display_name, description)

            tool_instance = Tool(
                name=tool_name,
                func=tool_cls(llm).get,
                description=description
            )
            loaded_tools.append(tool_instance)
            seen_names.add(tool_name)

    extra_tools = [
        ("Comprehensive Carcinogenicity Analysis", ComprehensiveCarcinogenicityTool(llm)._run, "Performs a comprehensive carcinogenicity analysis combining computational predictions and literature evidence for chemical compounds."),
        ("Comprehensive Mutagenicity Analysis", ComprehensiveMutagenicityTool(llm)._run, "Performs a comprehensive mutagenicity analysis combining computational predictions and literature evidence for chemical compounds."),
        ("Comprehensive Reproductive Toxicity Analysis", ComprehensiveReproductiveToxicityTool(llm)._run, "Performs a comprehensive reproductive toxicity analysis combining computational predictions and literature evidence for chemical compounds."),
        ("CMR Prediction", CMRPrediction()._run, "Predicts carcinogenicity, mutagenicity, and reproductive toxicity of molecules."),
        ("GHSClassification", GHSClassificationTool()._run, "Provides GHS classification for a compound."),
        ("CASRetriever", CASRetriever().get, "Retrieves CAS registry number of a compound."),
        ("SMILESRetriever", SMILESRetriever().get, "Retrieves SMILES representation of a compound."),
        ("NameRetriever", NameRetriever().get, "Retrieves chemical names and synonyms of a compound."),
        ("InChIRetriever", InChIRetriever().get, "Retrieves InChI string of a compound."),
        ("InChIKeyRetriever", InChIKeyRetriever().get, "Retrieves InChIKey of a compound."),
        ("FormulaRetriever", FormulaRetriever().get, "Retrieves molecular formula (Hill system) of a compound."),
        ("SymptomsTool", SymptomsTool(llm)._run, "Summarizes symptoms caused by exposure to the chemical (ingestion, skin contact, various routes)."),
        ("AcuteToxicityTool", AcuteToxicityTool(llm)._run, "Summarizes acute toxicity information for a compound."),
        ("HumanToxicityTool", HumanToxicityTool(llm)._run, "Summarizes human toxicity information for a compound."),
        ("EnvironmentalConcentrationsTool", EnvironmentalConcentrationsTool(llm)._run, "Provides information on environmental concentrations of a compound."),
        ("NonHumanToxicityTool", NonHumanToxicityTool(llm)._run, "Summarizes non-human toxicity information for a compound."),
        ("SafeStorageTool", SafeStorageTool(llm)._run, "Summarizes safe storage measures, conditions, and requirements for a chemical."),
        ("HazardsSummaryTool", HazardsSummaryTool(llm)._run, "Summarizes hazards overview and health hazards for a chemical."),
        ("EmergencyResponseMeasuresTool", EmergencyResponseMeasuresTool(llm)._run, "Summarizes nonfire spill response, DOT emergency guidelines, antidote and emergency treatment, and ERPGs for a chemical."),
    ]

    for name, func, desc in extra_tools:
        if name not in seen_names:
            loaded_tools.append(Tool(name=name, func=func, description=desc))
            seen_names.add(name)

    if verbose:
        print(f"Total tools loaded: {len(loaded_tools)}")

    return loaded_tools
