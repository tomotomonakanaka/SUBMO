import rdkit

def get_natom(mol):
    d_mol_h = rdkit.Chem.AddHs(mol)
    return len(d_mol_h.GetAtoms())

def get_nelec(mol):
    nelec = 0
    d_mol_h = rdkit.Chem.AddHs(mol)
    for atom in d_mol_h.GetAtoms():
        nelec += atom.GetAtomicNum()
    return nelec - rdkit.Chem.rdmolops.GetFormalCharge(mol)