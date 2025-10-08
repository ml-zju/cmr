import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Optional, Union
from agents.agent import CMR
from tools import load_llm_tools
from llm.utils import get_llm
from rdkit import Chem
from tools.PhysicalProperties import PROPERTY_TOOLS, create_property_tool
from tools.healthInformation import HEALTH_EFFECT_TOOLS, create_health_tool
from tools.EnvironmentalInformation import ENVIRONMENTAL_PROPERTY_TOOLS, create_environment_tool
import matplotlib.pyplot as plt
import time

start_time = time.time()

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
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False, "Invalid SMILES string."
        return True, ""
    except Exception as e:
        return False, f"Error validating SMILES: {str(e)}"


def load_all_tools(llm) -> Dict:
    tools = {
        "physical": load_property_tools(llm),
        "environmental": load_environmental_tools(llm),
        "health": load_health_tools(llm)
    }
    return tools


def load_property_tools(llm) -> Dict:
    property_tools = {}
    for tool_name, (display_name, description) in PROPERTY_TOOLS.items():
        try:
            ToolClass = create_property_tool(display_name, description)
            property_tools[tool_name] = ToolClass(llm)
        except Exception as e:
            st.warning(f"Failed to load property tool {tool_name}: {str(e)}")
    return property_tools


def load_environmental_tools(llm) -> Dict:
    environmental_tools = {}
    for tool_name, (display_name, description) in ENVIRONMENTAL_PROPERTY_TOOLS.items():
        try:
            ToolClass = create_environment_tool(display_name, description)
            environmental_tools[tool_name] = ToolClass(llm)
        except Exception as e:
            st.warning(f"Failed to load environmental tool {tool_name}: {str(e)}")
    return environmental_tools


def load_health_tools(llm) -> Dict:
    health_tools = {}
    for tool_name, (display_name, description) in HEALTH_EFFECT_TOOLS.items():
        try:
            ToolClass = create_health_tool(display_name, description)
            health_tools[tool_name] = ToolClass(llm)
        except Exception as e:
            st.warning(f"Failed to load health tool {tool_name}: {str(e)}")
    return health_tools


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
            "Analyze the environmental impact and behavior of this compound systematically:\n"
            "1. **Bioaccumulation Potential**: Assess the compound's tendency to accumulate in organisms\n"
            "2. **Biodegradation Characteristics**: Evaluate how readily the compound breaks down in the environment\n"
            "3. **Environmental Fate**: Describe the compound's behavior and distribution in environmental compartments\n"
            "4. **Ecotoxicological Concerns**: Identify potential risks to aquatic and terrestrial organisms\n"
            "5. **Overall Environmental Risk Assessment**: Provide a comprehensive evaluation of environmental hazards\n\n"
            "Structure your response with clear numbered points addressing each aspect above. "
            "Use specific data from the properties when available."
        ),
        "cmr": (
            f"Based on the following CMR predictions:\n{base_data}"
            "Predict the carcinogenic, mutagenic, and reproductive toxicity risks. "
            "Don't Explain the significance of these predictions for human safety. "
            "Identify which aspects pose the most significant concerns. "
            "Summarize in 3-5 sentences."
        ),
        "health": (
            f"Based on the following toxicological and health data:\n{base_data}"
            "Provide a systematic toxicological assessment structured as follows:\n"
            "1. **CMR Validation**: Compare the available toxicological evidence with CMR predictions and explain consistency or discrepancies\n"
            "2. **Primary Health Hazards**: Identify the most critical toxic effects and target organs\n"
            "3. **Exposure Route Analysis**: Evaluate risks associated with different exposure pathways (inhalation, dermal, oral)\n"
            "4. **Acute vs. Chronic Effects**: Distinguish between immediate and long-term health impacts\n"
            "5. **Vulnerable Populations**: Identify groups at higher risk (if applicable)\n"
            "6. **Risk Characterization**: Provide an overall assessment of the toxicological profile\n\n"
            "Structure your response with clear numbered points addressing each aspect above. "
            "Use specific toxicological data when available."
        )
    }

    return prompts.get(category, "")


def perform_prediction(model: CMR, smiles: str) -> Optional[Dict]:
    progress_bar = st.progress(0)
    try:
        results = {}
        total_steps = len(PROPERTY_CATEGORIES)

        molecule_info_tools = ["CASRetriever", "NameRetriever", "FormulaRetriever"]
        for tool_name in molecule_info_tools:
            try:
                for tool in model.tools:
                    if tool.name == tool_name:
                        results[tool_name] = tool.run(smiles)
                        break
                else:
                    results[tool_name] = "Not Available"
            except Exception as e:
                st.warning(f"Error getting {tool_name}: {str(e)}")
                results[tool_name] = "Not Available"

        if 'all_tools' not in st.session_state:
            st.session_state.all_tools = load_all_tools(model.llm)
        all_tools = st.session_state.all_tools

        cmr_short_map = {
            "Carcinogenic": "C",
            "Mutagenic": "M",
            "Toxic to Reproduction": "R"
        }

        for i, (category, info) in enumerate(PROPERTY_CATEGORIES.items()):
            category_data = []
            category_tools = all_tools.get(category, {})

            for property_name in info["properties"]:
                try:
                    if category == "cmr":
                        cmr_tool = next((t for t in model.tools if t.name == "CMR Prediction"), None)
                        if cmr_tool:
                            short = cmr_short_map.get(property_name)
                            if not short:
                                st.warning(f"No short mapping for property: {property_name}")
                                continue

                            tool_input = json.dumps({
                                'smiles': [smiles],
                                'property_type': short
                            })

                            try:
                                results_df = cmr_tool.run(tool_input)

                                if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                                    first_row = results_df.iloc[0]
                                    if first_row.get('error') is not None:
                                        st.error(f"CMR tool error for {property_name}: {first_row['error']}")
                                        results[f"{property_name}_Prediction"] = "Tool Error"
                                        category_data.append(f"{property_name}: Tool Error")
                                        continue

                                    label = first_row.get('label')
                                    if label is None:
                                        st.warning(f"No prediction available for {property_name}")
                                        results[f"{property_name}_Prediction"] = "No Prediction"
                                        category_data.append(f"{property_name}: No Prediction")
                                        continue

                                    # 获取其他字段
                                    confidence = first_row.get('calibrated_confid', 'N/A')
                                    in_domain = first_row.get('in_domain', 'N/A')

                                    prediction = 'Positive' if label == 1 else 'Negative'

                                    results[f"{property_name}_Prediction"] = {
                                        'label': int(label),
                                        'confidence': float(confidence) if confidence != 'N/A' and confidence is not None else 'N/A',
                                        'in_domain': bool(in_domain) if in_domain != 'N/A' and in_domain is not None else 'N/A',
                                        'prediction': prediction
                                    }

                                    conf_str = f"{confidence:.3f}" if confidence != 'N/A' and confidence is not None else 'N/A'
                                    domain_str = str(in_domain) if in_domain != 'N/A' and in_domain is not None else 'N/A'
                                    pred_str = f"{prediction} (Conf: {conf_str}, In-Domain: {domain_str})"
                                    category_data.append(f"{property_name}: {pred_str}")

                                else:
                                    st.warning(f"No valid results for {property_name}")
                                    results[f"{property_name}_Prediction"] = "Not Available"
                                    category_data.append(f"{property_name}: Not Available")

                            except Exception as e:
                                st.error(f"Error in CMR prediction for {property_name}: {str(e)}")
                                results[f"{property_name}_Prediction"] = "Error"
                                category_data.append(f"{property_name}: Error")
                        else:
                            st.warning("CMR Prediction tool not found")
                            results[f"{property_name}_Prediction"] = "Tool Not Available"
                            category_data.append(f"{property_name}: Tool Not Available")
                    elif property_name in category_tools:
                        tool = category_tools[property_name]
                        result = tool.run(smiles)
                        results[property_name] = result
                        if category not in ["environmental", "health"]:
                            category_data.append(f"{property_name}: {result}")
                    else:
                        results[property_name] = "Not Available"
                        if category not in ["environmental", "health"]:
                            category_data.append(f"{property_name}: Not Available")
                except Exception as e:
                    st.warning(f"Error getting {property_name}: {str(e)}")
                    results[property_name] = "Not Available"
                    if category not in ["environmental", "health"]:
                        category_data.append(f"{property_name}: Not Available")

            if category in ["environmental", "health"]:
                summary_data = []
                for property_name in info["properties"]:
                    result = results.get(property_name, "Not Available")
                    if result and result != "Not Available":
                        summary_data.append(f"{property_name}: {result}")

                summary_prompt = (
                    get_summary_prompt(category, summary_data, info)
                    if summary_data else
                    f"No specific {category} data available for this compound. Provide a general assessment based on structural considerations."
                )
            else:
                summary_prompt = (
                    get_summary_prompt(category, category_data, info)
                    if category_data else
                    f"No specific {category} data available for this compound."
                )

            try:
                category_summary = model.llm.predict(summary_prompt)
                results[f"{category}_summary"] = category_summary
            except Exception as e:
                st.error(f"Error generating {category} summary: {str(e)}")
                results[f"{category}_summary"] = "Summary generation failed"

            progress_bar.progress((i + 1) / total_steps)

        return results

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.exception(e)
        return None
    finally:
        progress_bar.empty()

def display_results(results, smiles):
    st.title("Comprehensive Prediction Report of CMR")
    st.markdown("*A Systematic Analysis of Physicochemical, Toxicological, and Environmental Properties*")
    st.markdown("---")

    st.header("I. Names and Identifiers")
    molecular_data = {
        "Identifiers": ["IUPAC Name", "CAS Number", "Molecular Formula", "SMILES"],
        "Name": [
            results.get('NameRetriever', 'Not Available'),
            results.get('CASRetriever', 'Not Available'),
            results.get('FormulaRetriever', 'Not Available'),
            smiles
        ]
    }
    df = pd.DataFrame(molecular_data)
    st.markdown(df.to_html(index=False), unsafe_allow_html=True)

    section_mapping = {
        "physical": "II. Physicochemical Characterization",
        "environmental": "III. Environmental Fate and Behavior Analysis",
        "cmr": "IV. Carcinogenic, Mutagenic, and Reproductive Toxicity Prediction",
        "health": "V. Toxicological Profile and Health Hazard Evaluation"
    }

    st.markdown("""
        <style>
            .dataframe {
                font-size: 14px;
                width: 100%;
                border-collapse: collapse;
            }
            .dataframe th {
                background-color: #f0f2f6;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                border: 1px solid #e1e4e8;
            }
            .dataframe td {
                padding: 10px;
                border: 1px solid #e1e4e8;
            }
            .dataframe tr:nth-child(even) {
                background-color: #f8f9fa;
            }
        </style>
        """, unsafe_allow_html=True)

    for category, info in PROPERTY_CATEGORIES.items():
        st.markdown("---")
        st.header(section_mapping[category])

        if category in ["environmental", "health"]:
            display_critical_analysis(results, category)
        else:
            with st.expander("View Detailed Data Table", expanded=True):
                if category == "cmr":
                    display_cmr_results_from_dict(info, results)
                else:
                    display_other_results(info, results)

            display_critical_analysis(results, category)

    display_methodology_section()
    display_report_footer()


def display_cmr_results_from_dict(info, results):
    cmr_data = []

    for prop in info["properties"]:
        prediction_key = f"{prop}_Prediction"
        result = results.get(prediction_key, {})

        if isinstance(result, dict) and 'prediction' in result:
            confidence = result.get('confidence', 'N/A')
            in_domain = result.get('in_domain', 'N/A')
            prediction = result.get('prediction', 'Not Available')

            color = get_prediction_color(confidence if isinstance(confidence, (int, float)) else 0)

            cmr_data.append({
                "Endpoint": prop,
                "Confidence": f"{confidence:.3f}" if isinstance(confidence, (int, float)) else str(confidence),
                "In-Domain": str(in_domain),
                "Prediction": f"<span style='color: {color}; font-weight: bold;'>{prediction}</span>"
            })
        else:
            cmr_data.append({
                "Endpoint": prop,
                "Confidence": "N/A",
                "In-Domain": "N/A",
                "Prediction": "<span style='color: gray;'>Not Available</span>"
            })

    if cmr_data:
        df_category = pd.DataFrame(cmr_data)
        st.markdown(df_category.to_html(escape=False, index=False), unsafe_allow_html=True)
        display_prediction_guide()
    else:
        st.warning("No CMR prediction results available.")


def get_prediction_color(prob):
    try:
        if isinstance(prob, str):
            if prob == 'N/A':
                return "gray"
            prob = float(prob)

        if prob >= 0.7:
            return "red"
        elif prob >= 0.3:
            return "orange"
        else:
            return "green"
    except (ValueError, TypeError):
        return "gray"


def display_other_results(info, results):
    filtered_properties = [prop for prop in info["properties"] if results.get(prop, '') != 'No information found']
    filtered_values = [results.get(prop, 'Not Available') for prop in filtered_properties]

    properties_data = {
        "Parameter": filtered_properties,
        "Determined Value": filtered_values
    }

    if properties_data["Parameter"]:
        df_category = pd.DataFrame(properties_data)
        st.markdown(df_category.to_html(index=False), unsafe_allow_html=True)
    else:
        st.warning("No data available for this category.")


def display_prediction_guide():
    st.markdown("""
    **Prediction Guide:**
    - 🔴 **Red**: High risk (Probability ≥ 0.7)
    - 🟠 **Orange**: Moderate risk (0.3 ≤ Probability < 0.7)
    - 🟢 **Green**: Low risk (Probability < 0.3)
    """)


def display_critical_analysis(results, category):
    analysis_titles = {
        "physical": "Physicochemical Properties Summary",
        "environmental": "Environmental Impact Assessment",
        "cmr": "CMR Hazard Evaluation",
        "health": "Toxicological Risk Assessment"
    }

    st.subheader(analysis_titles.get(category, "Analysis"))
    if f"{category}_summary" in results:
        summary = results[f"{category}_summary"]
        st.markdown(f"""
            <div style='background-color: #f8f9fa; 
                     padding: 20px; 
                     border-radius: 8px; 
                     border-left: 5px solid #0366d6;
                     font-size: 14px;
                     line-height: 1.8;
                     margin: 10px 0;'>
                {summary}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning(f"Summary not available for {category} category.")


def display_methodology_section():
    st.markdown("---")
    st.header("VI. Methodology and Limitations")
    with st.expander("📋 Technical Notes and Methodological Considerations"):
        st.markdown("""
        **Analytical Methodology:**
        - Computational predictions based on CMRScreen
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


def display_report_footer():
    st.markdown("---")
    st.caption(f"""
        Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
        Analysis Protocol Version: 2.0  
        Computational Assessment Framework: CMRScreen
    """)


def save_report_to_file(results, smiles):
    section_mapping = {
        "physical": "II. Physicochemical Characterization",
        "environmental": "III. Environmental Fate and Behavior Analysis",
        "cmr": "IV. Carcinogenic, Mutagenic, and Reproductive Toxicity Prediction",
        "health": "V. Toxicological Profile and Health Hazard Evaluation"
    }

    analysis_titles = {
        "physical": "Physicochemical Properties Summary",
        "environmental": "Environmental Impact Assessment",
        "cmr": "CMR Hazard Evaluation",
        "health": "Toxicological Risk Assessment"
    }

    report_content = []

    report_content.append("# Comprehensive Prediction Report of CMR")
    report_content.append("*A Systematic Analysis of Physicochemical, Toxicological, and Environmental Properties*")
    report_content.append("\n---\n")

    report_content.append("## I. Names and Identifiers")
    report_content.append(f"IUPAC Name: {results.get('NameRetriever', 'Not Available')}")
    report_content.append(f"CAS Number: {results.get('CASRetriever', 'Not Available')}")
    report_content.append(f"Molecular Formula: {results.get('FormulaRetriever', 'Not Available')}")
    report_content.append(f"SMILES: {smiles}")

    for category, info in PROPERTY_CATEGORIES.items():
        report_content.append(f"\n## {section_mapping[category]}")

        if category not in ["environmental", "health"]:
            report_content.append("\n### Detailed Results")
            for prop in info["properties"]:
                if category == "cmr":
                    prediction_key = f"{prop}_Prediction"
                    result = results.get(prediction_key, {})
                    if isinstance(result, dict) and 'prediction' in result:
                        confidence = result.get('confidence', 'N/A')
                        in_domain = result.get('in_domain', 'N/A')
                        prediction = result.get('prediction', 'Not Available')

                        report_content.append(f"**{prop}:**")
                        report_content.append(f"- Confidence: {confidence:.3f}" if isinstance(confidence, (int, float)) else f"- Confidence: {confidence}")
                        report_content.append(f"- In-Domain: {in_domain}")
                        report_content.append(f"- Prediction: {prediction}")
                else:
                    value = results.get(prop, "Not Available")
                    report_content.append(f"**{prop}:** {value}")

        if f"{category}_summary" in results:
            report_content.append(f"\n### {analysis_titles[category]}")
            report_content.append(results[f"{category}_summary"])

    report_content.append("\n## VI. Methodology and Limitations")
    report_content.append("""
### Technical Notes and Methodological Considerations

**Analytical Methodology:**
- Computational predictions based on CMRScreen
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
Computational Assessment Framework: CMRScreen
    """)

    report_text = "\n".join(report_content)
    st.download_button(
        label="📄 Download Complete Report",
        data=report_text,
        file_name=f"cmr_prediction_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )


def generate_report_main(
        model_name: str,
        temperature: float,
        api_key: str,
        **kwargs
):
    if 'model' not in st.session_state:
        try:
            llm = get_llm(
                model_name=model_name,
                temperature=temperature,
                api_key=api_key,
                **kwargs
            )
            tools = load_llm_tools(llm, verbose=True)
            st.session_state.model = CMR(llm=llm, tools=tools)
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            st.exception(e)
            return

    col1, col2 = st.columns(2)
    with col1:
        smiles = st.text_input("Enter SMILES string:", placeholder="e.g., CCO")
        is_valid, error_message = validate_smiles(smiles)
        if not is_valid and smiles:
            st.error(error_message)

    if st.button("🚀 Generate Comprehensive Report", type="primary") and is_valid:
        with st.spinner("Generating comprehensive analysis..."):
            results = perform_prediction(st.session_state.model, smiles)
            if results:
                display_results(results, smiles)
                save_report_to_file(results, smiles)

    with st.expander("ℹ️ Help & Information"):
        st.markdown("""
        **How to use this tool:**
        1. Enter a valid SMILES string for your compound
        2. Click 'Generate Comprehensive Report' to see the analysis
        3. Download the complete report when analysis is finished

        **SMILES Examples:**
        - Ethanol: `CCO`
        - Benzene: `c1ccccc1`
        - Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`

        **Note:** 
        - Some properties might not be available for all compounds
        - Environmental and health sections show structured summaries instead of raw data
        - CMR predictions include probability scores for risk assessment
        """)

    end_time = time.time()
    execution_time = end_time - start_time
    st.info(f"⏱️ Total execution time: {execution_time:.2f} seconds")