import os
import torch
import numpy as np
import joblib
from chemprop import data, featurizers
from chemprop.featurizers.molgraph.cache import MolGraphCacheOnTheFly
from model import MPNN
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def check_AD(smiles: str, pca_fp_file_path: str, threshold_percentile: float = 99) -> bool:
    try:
        if not os.path.exists(pca_fp_file_path):
            print(f"PCA fingerprint file not found: {pca_fp_file_path}")
            return False

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES string: {smiles}")
            return False

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        query_fp = np.zeros((1, 2048), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, query_fp[0])

        with open(pca_fp_file_path, 'rb') as f:
            pca_data = pickle.load(f)

        pca_model = pca_data['pca_model']
        pca_train = pca_data['pca_fingerprints']

        pca_query = pca_model.transform(query_fp)

        train_mean = np.mean(pca_train, axis=0)
        diff = pca_query[0] - train_mean
        distance = np.sqrt(np.sum(diff ** 2))

        train_distances = np.array([np.sqrt(np.sum((point - train_mean) ** 2))
                                    for point in pca_train])
        threshold = np.percentile(train_distances, threshold_percentile)

        return distance <= threshold

    except Exception as e:
        print(f"Error in AD check: {str(e)}")
        return False

def calibrate_prediction(prob, temperature):
    logit = np.log(prob / (1 - prob))
    calibrated_logit = logit / temperature
    return 1 / (1 + np.exp(-calibrated_logit))


def predict_cmr(smiles: str, task_name: str = 'C'):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Base directory: {base_dir}")

        ensemble_pth_path = os.path.join(base_dir, 'checkpoint', task_name, f'{task_name}.pth')
        temperature_model_path = os.path.join(base_dir, 'checkpoint', task_name, f'{task_name}.joblib')
        fingerprint_file_path = os.path.join(base_dir, 'AD', f'{task_name}_pca_fp.pkl')

        print(f"Ensemble model path: {ensemble_pth_path}")
        print(f"Temperature model path: {temperature_model_path}")
        print(f"PCA fingerprint path: {fingerprint_file_path}")

        temperature = joblib.load(temperature_model_path)

        ensemble_data = torch.load(ensemble_pth_path, map_location=torch.device('cpu'))
        models = []

        for path in ensemble_data['top_model_paths']:
            if os.path.isabs(path):
                model_path = path
            else:
                clean_path = path.lstrip('./\\')
                model_path = os.path.join(base_dir, clean_path)

            print(f"Loading model from: {model_path}")

            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                continue

            model = MPNN(n_classes=2)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)

        if not models:
            print("No valid models loaded!")
            return None, None, False

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        is_in_ad_domain = check_AD(smiles, fingerprint_file_path)
        if not is_in_ad_domain:
            print(f"The molecule {smiles} is out of the AD domain.")
            return None, None, False

        data_point = data.MoleculeDatapoint.from_smi(smiles)
        mol_graph = MolGraphCacheOnTheFly([data_point.mol], [None], [None], featurizer)[0]
        batched_graph = data.collate.BatchMolGraph([mol_graph])

        probs_class1 = []
        for model in models:
            with torch.no_grad():
                output = model(batched_graph, None, None)[:, 0]
                prob_softmax = torch.nn.functional.softmax(output, dim=1).detach().numpy()
                probs_class1.append(prob_softmax[0, 1])

        avg_prob = np.mean(probs_class1)

        calibrated_prob = calibrate_prediction(avg_prob, temperature)
        calibrated_confid = round(max(calibrated_prob, 1 - calibrated_prob), 4)
        label = int(calibrated_prob >= 0.5)

        return label, calibrated_confid, True

    except Exception as e:
        print(f"Error processing task '{task_name}' and SMILES '{smiles}': {e}")
        return None, None, False



