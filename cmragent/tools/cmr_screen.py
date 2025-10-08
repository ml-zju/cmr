from langchain.tools import BaseTool
import pandas as pd
from rdkit import Chem
import ast
from predictor import predict_cmr


class CMRPrediction(BaseTool):
    name: str = "CMR Prediction"
    description: str = (
        "Predicts carcinogenic, mutagenic, and reproductive toxicity properties of chemical compounds. "
        "Input must be a JSON string with the following format: "
        "{'smiles': ['SMILES_STRING1', 'SMILES_STRING2', ...], 'property_type': 'PROPERTY_TYPE'} "
        "where PROPERTY_TYPE must be one of: 'C', 'M', or 'R'. "
        "Example: {'smiles': ['CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O'], 'property_type': 'C'} for BPA carcinogenicity prediction. "
        "Chemical names or CAS numbers are NOT directly accepted - they must first be converted to SMILES notation."
    )
    return_direct: bool = True

    def validate_smiles(self, smiles: str) -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES string: {smiles}")
                return False
            return True
        except Exception as e:
            print(f"Error validating SMILES {smiles}: {e}")
            return False

    def normalize_property_type(self, property_type: str) -> str:
        property_mapping = {
            'carcinogenic': 'C',
            'carcinogenicity': 'C',
            'c': 'C',
            'mutagenic': 'M',
            'mutagenicity': 'M',
            'm': 'M',
            'reproductive': 'R',
            'reproductive toxicity': 'R',
            'toxic to r': 'R',
            'r': 'R'
        }

        normalized = property_mapping.get(property_type.lower(), property_type.upper())
        if normalized not in ['C', 'M', 'R']:
            raise ValueError(f"Invalid property_type: {property_type}. Must be one of: 'C', 'M', 'R'")
        return normalized

    def _run(self, query: str) -> pd.DataFrame:
        try:
            if not isinstance(query, str):
                raise ValueError("Input query must be a string representing a dictionary.")

            try:
                input_data = ast.literal_eval(query)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Invalid JSON format: {e}")

            if not isinstance(input_data, dict):
                raise ValueError("Input query must be a string representing a dictionary.")

            smiles_list = input_data.get('smiles', [])
            property_type = input_data.get('property_type', '')

            if not smiles_list or not isinstance(smiles_list, list):
                raise ValueError("Input must contain a 'smiles' key with a list of SMILES strings.")
            if not property_type or not isinstance(property_type, str):
                raise ValueError("Input must contain a 'property_type' key with a string value.")

            try:
                normalized_property_type = self.normalize_property_type(property_type)
            except ValueError as e:
                raise ValueError(str(e))

            valid_smiles = []
            for smiles in smiles_list:
                if self.validate_smiles(smiles):
                    valid_smiles.append(smiles)
                else:
                    print(f"Skipping invalid SMILES: {smiles}")

            if not valid_smiles:
                raise ValueError("No valid SMILES strings provided.")

            results = []
            for smiles in valid_smiles:
                try:
                    prediction_result = predict_cmr(smiles, normalized_property_type)

                    if prediction_result is None:
                        results.append({
                            'smiles': smiles,
                            'label': None,
                            'calibrated_confid': None,
                            'in_domain': False,
                            'error': 'Prediction failed'
                        })
                    elif isinstance(prediction_result, tuple) and len(prediction_result) == 3:
                        label, calibrated_confid, in_domain = prediction_result
                        results.append({
                            'smiles': smiles,
                            'label': label,
                            'calibrated_confid': calibrated_confid,
                            'in_domain': in_domain,
                            'error': None if label is not None else 'Prediction failed'
                        })
                    else:
                        results.append({
                            'smiles': smiles,
                            'label': None,
                            'calibrated_confid': None,
                            'in_domain': False,
                            'error': f'Unexpected return format: {type(prediction_result)}'
                        })

                except Exception as e:
                    print(f"Error predicting for SMILES {smiles}: {e}")
                    results.append({
                        'smiles': smiles,
                        'label': None,
                        'calibrated_confid': None,
                        'in_domain': False,
                        'error': str(e)
                    })

            results_df = pd.DataFrame(results)

            return results_df

        except Exception as e:
            print(f"Error in _run: {str(e)}")
            error_df = pd.DataFrame([{
                'smiles': 'N/A',
                'label': None,
                'calibrated_confid': None,
                'in_domain': False,
                'error': str(e),
            }])
            return error_df

    def _interpret_result(self, row, property_type):
        if row['error'] is not None:
            return f"Error: {row['error']}"

        if not row['in_domain']:
            return "Compound is outside the applicability domain - prediction may be unreliable"

        if row['label'] is None:
            return "Prediction failed"

        property_names = {
            'C': 'carcinogenic',
            'M': 'mutagenic',
            'R': 'reproductive toxic'
        }

        property_name = property_names.get(property_type, 'toxic')

        if row['label'] == 1:
            result = f"Predicted to be {property_name}"
        else:
            result = f"Predicted to be non-{property_name}"

        confidence = row['calibrated_confid']
        if confidence is not None:
            result += f" (confidence: {confidence:.3f})"

        return result

    async def _arun(self, query: str) -> pd.DataFrame:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, query)


