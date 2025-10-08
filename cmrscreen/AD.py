import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, BondType, rdmolops
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors, rdchem, PandasTools, EState
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from rdkit import DataStructs

def compute_morgan_fingerprints(smiles_list, radius=2, nBits=2048):
    """Generate Morgan fingerprints for a list of SMILES."""
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            fps.append(fp)
    return fps

def convert_fingerprints_to_numpy(fps):
    """Convert RDKit fingerprints to numpy array."""
    num_fps = len(fps)
    fp_length = len(fps[0])
    np_fps = np.zeros((num_fps, fp_length), dtype=np.int8)
    
    for i, fp in enumerate(fps):
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        np_fps[i] = arr
    
    return np_fps
    
def analyze_AD(train_data, test_data, variance_threshold=0.90,
                               threshold_percentile=99):
    """Analyze applicability domain using PCA and Euclidean distance."""
    # Initialize and fit PCA
    pca_full = PCA()
    pca_full.fit(train_data)
    
    # Find number of components for target variance
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= variance_threshold) + 1
    print(f"\nUsing {n_components} components to explain {variance_threshold*100}% of variance")
    
    # Perform PCA with determined components
    pca = PCA(n_components=n_components)
    pca_train = pca.fit_transform(train_data)
    pca_test = pca.transform(test_data)
    
    # Calculate Euclidean distances
    train_mean = np.mean(pca_train, axis=0)
    
    def calculate_distance(point):
        diff = point - train_mean
        return np.sqrt(np.sum(diff**2))
    
    train_distances = np.array([calculate_distance(point) for point in pca_train])
    test_distances = np.array([calculate_distance(point) for point in pca_test])
    
    # Define threshold and classify compounds
    threshold = np.percentile(train_distances, threshold_percentile)
    train_out_indices = train_distances > threshold
    test_out_indices = test_distances > threshold
    
    # Print detailed training distances information
    print(f"\nTraining distances statistics:")
    print(f"99th percentile threshold: {threshold:.3f} (This means 99% of training compounds have distances below this value)")
    print(f"Training set - Min distance: {np.min(train_distances):.3f}")
    print(f"Training set - Max distance: {np.max(train_distances):.3f}")
    print(f"Training set - Mean distance: {np.mean(train_distances):.3f}")
    print(f"Training set - Median distance: {np.median(train_distances):.3f}")
    
    print(f"\nTest set distances statistics:")
    print(f"Test set - Min distance: {np.min(test_distances):.3f}")
    print(f"Test set - Max distance: {np.max(test_distances):.3f}")
    print(f"Test set - Mean distance: {np.mean(test_distances):.3f}")
    print(f"Test set - Median distance: {np.median(test_distances):.3f}")
    
    # Print counts above threshold
    n_train_above = sum(train_distances > threshold)
    n_test_above = sum(test_distances > threshold)
    print(f"\nCompounds above threshold:")
    print(f"Training set: {n_train_above} ({(n_train_above/len(train_distances)*100):.1f}% of training set)")
    print(f"Test set: {n_test_above} ({(n_test_above/len(test_distances)*100):.1f}% of test set)")
    
    # Print summary statistics
    print("\nApplicability Domain Analysis Results")
    print("-" * 50)
    print(f"Training set: {sum(train_out_indices)} compounds out of {len(train_data)} ({sum(train_out_indices)/len(train_data)*100:.1f}%) out of domain")
    print(f"Test set: {sum(test_out_indices)} compounds out of {len(test_data)} ({sum(test_out_indices)/len(test_data)*100:.1f}%) out of domain")

    return {
        'train_out_indices': train_out_indices,
        'test_out_indices': test_out_indices,
        'threshold': threshold,
        'pca': pca,
        'explained_variance': sum(pca.explained_variance_ratio_),
        'n_components': n_components
    }

# 读取数据
data = pd.read_csv("/root/chemprop/dataset/C.csv")

# 根据Seed_0列的标记来分配训练集和测试集
train_data = data[data['Seed_0'] == 'train']
test_data = data[data['Seed_0'] == 'test']


from AD import analyze_AD
print("\nGenerating Morgan fingerprints...")
train_fps = compute_morgan_fingerprints(train_data['smiles'])
test_fps = compute_morgan_fingerprints(test_data['smiles'])

# Convert to numpy arrays
train_fp_array = convert_fingerprints_to_numpy(train_fps)
test_fp_array = convert_fingerprints_to_numpy(test_fps)

print("\nAnalyzing applicability domain with Morgan fingerprints...")
print("\nTest set analysis:")
morgan_test_results = analyze_AD(train_fp_array, test_fp_array)