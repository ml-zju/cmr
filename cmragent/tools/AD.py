import pickle
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from rdkit import DataStructs

def check_AD(smiles: str, fp_file_path: str, variance_threshold: float = 0.90, threshold_percentile: float = 99) -> bool:
    try:
        if not os.path.exists(fp_file_path):
            print(f"Fingerprint file not found: {fp_file_path}")
            return False

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES string: {smiles}")
            return False

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        query_fp = np.zeros((1, 2048), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, query_fp[0])

        with open(fp_file_path, 'rb') as f:
            train_fp_array = pickle.load(f)

        pca_full = PCA()
        pca_full.fit(train_fp_array)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= variance_threshold) + 1

        pca = PCA(n_components=n_components)
        pca_train = pca.fit_transform(train_fp_array)
        pca_query = pca.transform(query_fp)

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