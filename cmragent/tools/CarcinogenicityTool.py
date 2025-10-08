import asyncio
import requests
from typing import Dict, Any, Optional, List
from langchain.tools import BaseTool
from langchain.llms.base import LLM
from langchain import LLMChain, PromptTemplate
import pandas as pd
from rdkit import Chem

from tools.get_cid import get_compound_cid
from tools.cmr_screen import CMRPrediction
from tools.C_evidence import CarcinogenicityEvidenceTool


def is_smiles(text):
    """Check if text is a valid SMILES notation"""
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        if m is None:
            return False
        return True
    except:
        return False


class ComprehensiveCarcinogenicityTool(BaseTool):
    name: str = "ComprehensiveCarcinogenicityTool"
    description: str = (
        "Comprehensive carcinogenicity analysis tool that takes a chemical name, CAS number, or SMILES notation "
        "and provides: 1) Computational prediction of carcinogenicity, 2) Literature evidence from databases, "
        "and 3) An integrated summary combining both computational and experimental evidence. "
        "Input: chemical identifier (name, CAS, or SMILES)"
    )

    # 在类定义中声明字段
    llm: LLM
    cmr_tool: CMRPrediction
    evidence_tool: CarcinogenicityEvidenceTool

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm: LLM, **kwargs):
        cmr_tool = CMRPrediction()
        evidence_tool = CarcinogenicityEvidenceTool(llm=llm)

        super().__init__(
            llm=llm,
            cmr_tool=cmr_tool,
            evidence_tool=evidence_tool,
            **kwargs
        )

    def _convert_to_smiles(self, identifier: str) -> str:
        """Convert chemical identifier to SMILES notation"""
        if is_smiles(identifier):
            return identifier

        try:
            cid = get_compound_cid(identifier)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES/JSON"

            response = requests.get(url)
            response.raise_for_status()

            data = response.json()
            smiles = data['PropertyTable']['Properties'][0]['SMILES']

            return smiles

        except Exception as e:
            raise ValueError(f"Could not convert {identifier} to SMILES: {str(e)}")

    def _predict_carcinogenicity(self, smiles: str) -> Dict[str, Any]:
        try:
            print(f"Step 2: Predicting carcinogenicity for SMILES: {smiles}")

            query = f"{{'smiles': ['{smiles}'], 'property_type': 'C'}}"

            prediction_df = self.cmr_tool._run(query)
            print(prediction_df)

            if isinstance(prediction_df, pd.DataFrame) and not prediction_df.empty:
                result = prediction_df.iloc[0].to_dict()

                return {
                    'success': True,
                    'prediction_data': result,
                    'smiles': smiles
                }
            else:
                return {
                    'success': False,
                    'error': "CMR prediction returned empty results",
                    'smiles': smiles
                }

        except Exception as e:
            return {
                'success': False,
                'error': f"Error in carcinogenicity prediction: {str(e)}",
                'smiles': smiles
            }

    def _get_evidence(self, identifier: str) -> Dict[str, Any]:
        try:
            print(f"Step 3: Retrieving carcinogenicity evidence for: {identifier}")

            evidence = self.evidence_tool._run(identifier)

            if evidence and not evidence.startswith("Error"):
                return {
                    'success': True,
                    'evidence': evidence,
                    'identifier': identifier
                }
            else:
                return {
                    'success': False,
                    'error': f"Failed to retrieve evidence: {evidence}",
                    'identifier': identifier
                }

        except Exception as e:
            return {
                'success': False,
                'error': f"Error retrieving evidence: {str(e)}",
                'identifier': identifier
            }

    def _generate_comprehensive_summary(self,
                                        smiles: str,
                                        original_identifier: str,
                                        prediction_result: Dict,
                                        evidence_result: Dict) -> str:
        try:
            print("Step 4: Generating comprehensive summary...")

            summary_data = {
                'compound_info': {
                    'original_identifier': original_identifier,
                    'smiles': smiles
                },
                'computational_prediction': {},
                'literature_evidence': {},
                'errors': []
            }

            if prediction_result.get('success'):
                pred_data = prediction_result.get('prediction_data', {})
                summary_data['computational_prediction'] = {
                    'label': pred_data.get('label'),
                    'confidence': pred_data.get('calibrated_confid'),
                    'in_domain': pred_data.get('in_domain'),
                    'prediction_available': True
                }
            else:
                summary_data['computational_prediction']['prediction_available'] = False
                summary_data['errors'].append(f"Prediction error: {prediction_result.get('error')}")

            if evidence_result.get('success'):
                summary_data['literature_evidence'] = {
                    'evidence_text': evidence_result.get('evidence'),
                    'evidence_available': True
                }
            else:
                summary_data['literature_evidence']['evidence_available'] = False
                summary_data['errors'].append(f"Evidence error: {evidence_result.get('error')}")

            prompt_template = PromptTemplate(
                template="""
                Please provide a comprehensive carcinogenicity assessment for the following compound based on both computational predictions and literature evidence.

                Compound Information:
                - Original Identifier: {original_identifier}
                - SMILES: {smiles}

                Computational Prediction Results:
                {prediction_section}

                Literature Evidence:
                {evidence_section}

                Errors Encountered:
                {errors_section}

                Please provide a structured summary that includes:
                1. Compound Identification: Brief description of the compound
                2. Computational Assessment: Interpretation of the prediction results
                3. Literature Evidence: Summary of experimental/epidemiological evidence
                4. Integrated Conclusion: Overall carcinogenicity assessment considering both sources

                Format the response clearly with headers and bullet points where appropriate.
                """,
                input_variables=["original_identifier", "smiles", "prediction_section", "evidence_section",
                                 "errors_section"]
            )

            if summary_data['computational_prediction'].get('prediction_available'):
                pred = summary_data['computational_prediction']
                label_text = "Carcinogenic" if pred['label'] == 1 else "Non-carcinogenic"
                confidence_text = f"{pred['confidence']:.3f}" if pred['confidence'] is not None else "Not available"
                domain_text = "Yes" if pred['in_domain'] else "No"

                prediction_section = f"""
                - Prediction: {label_text}
                - Confidence: {confidence_text}
                - Within applicability domain: {domain_text}
                """
            else:
                prediction_section = "Computational prediction not available due to errors."

            if summary_data['literature_evidence'].get('evidence_available'):
                evidence_section = summary_data['literature_evidence']['evidence_text']
            else:
                evidence_section = "Literature evidence not available due to errors."

            errors_section = "\n".join(summary_data['errors']) if summary_data['errors'] else "No errors encountered."

            llm_chain = LLMChain(prompt=prompt_template, llm=self.llm)
            summary = llm_chain.run({
                "original_identifier": summary_data['compound_info']['original_identifier'],
                "smiles": summary_data['compound_info']['smiles'],
                "prediction_section": prediction_section,
                "evidence_section": evidence_section,
                "errors_section": errors_section
            })

            return summary

        except Exception as e:
            return f"Error generating comprehensive summary: {str(e)}"

    def _run(self, identifier: str) -> str:
        try:
            print(f"Starting comprehensive carcinogenicity analysis for: {identifier}")
            print("=" * 60)

            print(f"Step 1: Converting '{identifier}' to SMILES...")
            try:
                smiles = self._convert_to_smiles(identifier)
                print(f"Successfully converted to SMILES: {smiles}")
            except ValueError as e:
                return f"Error in SMILES conversion: {str(e)}"

            prediction_result = self._predict_carcinogenicity(smiles)

            evidence_result = self._get_evidence(identifier)

            comprehensive_summary = self._generate_comprehensive_summary(
                smiles, identifier, prediction_result, evidence_result
            )

            print("=" * 60)
            print("Analysis completed!")

            return comprehensive_summary

        except Exception as e:
            error_msg = f"Error in comprehensive carcinogenicity analysis: {str(e)}"
            print(error_msg)
            return error_msg

    async def _arun(self, identifier: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, identifier)