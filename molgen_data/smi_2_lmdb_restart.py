import multiprocessing
from rdkit import Chem
import pickle
import lmdb
import numpy as np
import os
import time
from rdkit.Chem import AllChem
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

def gen_conf(args):
    smi, mol_id = args
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=1, pruneRmsThresh=1, maxAttempts=1000, useRandomCoords=False)
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
        mol = Chem.RemoveHs(mol)
        if mol.GetNumConformers() == 0:
            return None
        return {
            'coordinates': [np.array(mol.GetConformer(i).GetPositions()) for i in range(mol.GetNumConformers())],
            'atoms': [a.GetSymbol() for a in mol.GetAtoms()],
            'smi': Chem.MolToSmiles(mol),
            'IDs': mol_id
        }
    except Exception as e:
        return None

def read_smiles_file(smi_file):
    molecules = []
    with open(smi_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                smi, mol_id = parts[0], parts[1]
            else:
                smi, mol_id = parts[0], f"mol_{len(molecules)}"
            molecules.append((smi, mol_id))
    return molecules

def get_processed_ids(lmdb_path):
    if not os.path.exists(lmdb_path):
        return set(), 0
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
    processed = set()
    with env.begin() as txn:
        for key, value in txn.cursor():
            d = pickle.loads(value)
            processed.add(d['IDs'])
    env.close()
    return processed, len(processed)

def process_smi_file(smi_file, lmdb_path, n_cpu=32, batch_size=10000):
    subset = os.path.basename(smi_file).split('.')[0]
    molecules = read_smiles_file(smi_file)
    total_input = len(molecules)
    
    processed_ids, num = get_processed_ids(lmdb_path)
    if processed_ids:
        print(f'Resuming: found {num} existing entries in {lmdb_path}')
        molecules = [(s, m) for s, m in molecules if m not in processed_ids]
        print(f'{len(molecules)} molecules remaining out of {total_input}')
    else:
        print(f'Processing {total_input} molecules from {smi_file}')
    
    if not molecules:
        print('Nothing to process')
        return num
    
    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, 
                    readahead=False, meminit=False, map_size=1099511627776 * 2)
    
    batch = []
    start_time = time.time()
    total = len(molecules)
    
    with multiprocessing.Pool(n_cpu) as pool:
        for i, result in enumerate(pool.imap(gen_conf, molecules, chunksize=200), 1):
            if result is not None:
                result['subset'] = subset
                batch.append(result)
            
            if len(batch) >= batch_size:
                with env.begin(write=True) as txn:
                    for d in batch:
                        txn.put(str(num).encode('ascii'), pickle.dumps(d))
                        num += 1
                batch = []
            
            if i % 1000 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (total - i) / rate
                print(f'[{elapsed/3600:.1f}h] {i}/{total} ({i*100/total:.1f}%) | {num} written | {rate:.0f} mol/s | ETA: {eta/3600:.1f}h')
    
    if batch:
        with env.begin(write=True) as txn:
            for d in batch:
                txn.put(str(num).encode('ascii'), pickle.dumps(d))
                num += 1
    
    env.close()
    total_time = time.time() - start_time
    print(f'Finished: {num} molecules written to {lmdb_path} in {total_time/3600:.1f}h')
    return num

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('smi_file', type=str, help='SMILES file')
    parser.add_argument('lmdb_path', type=str, help='output lmdb path')
    parser.add_argument('--n_cpu', type=int, default=32, help='number of CPUs')
    args = parser.parse_args()

    process_smi_file(args.smi_file, args.lmdb_path, args.n_cpu)