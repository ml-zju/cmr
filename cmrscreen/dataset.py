import random
from typing import Iterable
from dataclasses import dataclass, field

import pandas as pd
import torch
import numpy as np

from rdkit import Chem
from rdkit.Chem import Mol
from chemprop import data, featurizers
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import Featurizer
from chemprop.featurizers.molgraph.cache import MolGraphCache, MolGraphCacheOnTheFly
from chemprop.featurizers.molgraph import CGRFeaturizer, SimpleMoleculeMolGraphFeaturizer

from SmilesEnumerator import SmilesEnumerator


def get_sl_data(csv_file, data_tag='smiles', label_tag='class', split_label='Split', split='train'):
    data_frame = pd.read_csv(csv_file)
    # Filter the dataset based on the split
    if split == 'train':
        data_frame = data_frame[data_frame[split_label] == 'train']
    elif split == 'val':
        data_frame = data_frame[data_frame[split_label] == 'val']
    elif split == 'test':
        data_frame = data_frame[data_frame[split_label] == 'test']

    smiles_list = []
    classes_list = []
    for smiles, label in zip(data_frame[data_tag].tolist(), data_frame[label_tag].tolist()):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles_list.append(smiles)
                classes_list.append(label)
        except Exception as e:
            print(f"Warning: Failed to convert SMILES to mol: {smiles}. Error: {str(e)}")

    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles_list, classes_list)]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    return data.MoleculeDataset(all_data, featurizer)


class SSLMoleculeDataset(data.MoleculeDataset):
    data: list[str]
    featurizer: Featurizer[Mol, MolGraph] = field(default_factory=SimpleMoleculeMolGraphFeaturizer)

    def __post_init__(self):
        self.sme = SmilesEnumerator()

    def weak_augment(self, smiles):
        if random.random() < 0.2:
            return self.sme.randomize_smiles(smiles)
        return smiles

    def __getitem__(self, idx: int) -> data.Datum:
        smiles_w = self.data[idx]
        
        smiles_w = self.weak_augment(smiles_w)
        
        data_point_w = data.MoleculeDatapoint.from_smi(smiles_w)
        mg_w = MolGraphCacheOnTheFly([data_point_w.mol], [None], [None], self.featurizer)[0]
        data_w = data.Datum(mg_w, None, None, None, data_point_w.weight, None, None)

        smiles_s = self.sme.randomize_smiles(smiles_w) 
        #smiles_s = self.cyclize_molecule(smiles_s) 
        data_point_s = data.MoleculeDatapoint.from_smi(smiles_s)
        data_point_s.mol = self.random_delete_subgraph(data_point_s.mol)
        mg_s = MolGraphCacheOnTheFly([data_point_s.mol], [None], [None], self.featurizer)[0]
        data_s = data.Datum(mg_s, None, None, None, data_point_s.weight, None, None)

        return data_w, data_s

    def cyclize_molecule(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        atoms_to_cycle = [0, 1]
        new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True, rootedAtAtom=atoms_to_cycle[0]))
        new_smiles = Chem.MolToSmiles(new_mol)
        return new_smiles

    def random_delete_subgraph(self, mol):
        if mol.GetNumAtoms() < 2:
            return mol

        atom_to_delete = random.randint(0, mol.GetNumAtoms() - 1)
        new_mol = Chem.EditableMol(mol)
        new_mol.RemoveAtom(atom_to_delete)
        new_mol = new_mol.GetMol()
        if new_mol.GetNumAtoms() == 0:
            return mol
        return new_mol


def get_ssl_data(csv_file, data_tag='smiles'):
    data_frame = pd.read_csv(csv_file)
    smiles_list = []

    for smiles in data_frame[data_tag].tolist():
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles_list.append(smiles)
        except Exception as e:
            print(f"Warning: Failed to convert SMILES to mol: {smiles}. Error: {str(e)}")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    return SSLMoleculeDataset(smiles_list, featurizer)


def collate_batch(batch) -> data.collate.TrainingBatch:
    batch_w, batch_s = zip(*batch)
    mgs_w, V_ds_w, x_ds_w, ys_w, weights_w, lt_masks_w, gt_masks_w = zip(*batch_w)
    mgs_s, V_ds_s, x_ds_s, ys_s, weights_s, lt_masks_s, gt_masks_s = zip(*batch_s)

    return data.collate.TrainingBatch(
        data.collate.BatchMolGraph(mgs_w),
        None if V_ds_w[0] is None else torch.from_numpy(np.concatenate(V_ds_w)).float(),
        None if x_ds_w[0] is None else torch.from_numpy(np.array(x_ds_w)).float(),
        None if ys_w[0] is None else torch.from_numpy(np.array(ys_w)).float(),
        torch.tensor(weights_w, dtype=torch.float).unsqueeze(1),
        None if lt_masks_w[0] is None else torch.from_numpy(np.array(lt_masks_w)),
        None if gt_masks_w[0] is None else torch.from_numpy(np.array(gt_masks_w)),
    ), data.collate.TrainingBatch(
        data.collate.BatchMolGraph(mgs_s),
        None if V_ds_s[0] is None else torch.from_numpy(np.concatenate(V_ds_s)).float(),
        None if x_ds_s[0] is None else torch.from_numpy(np.array(x_ds_s)).float(),
        None if ys_s[0] is None else torch.from_numpy(np.array(ys_s)).float(),
        torch.tensor(weights_s, dtype=torch.float).unsqueeze(1),
        None if lt_masks_s[0] is None else torch.from_numpy(np.array(lt_masks_s)),
        None if gt_masks_s[0] is None else torch.from_numpy(np.array(gt_masks_s)),
    )
