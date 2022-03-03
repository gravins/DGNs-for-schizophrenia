from sklearn.model_selection import train_test_split
from rdkit.Chem import AllChem
from rdkit import Chem
import pandas as pd
import numpy as np
import argparse
import pickle
import time
import os


def create_compound_df(path="./Datasets/torch_dataset.csv"):
    # Read dataset
    dict_path = {"qHTS": {"path": "./Spark_MEGF10_qHTS_explained.xlsx", "sheet": ["LOPAC Results", "MS_BM_Results", "Well Data"]},
                 "NIHCC": {"path": "./Spark_MEGF10_NIHCC_1.xlsx", "sheet": ["NIHCC Sum"]}}
    ft = ["STF_ID", "uM", "%Pos  W2/Med", "%Inh Cells/High"]
    additional_ft = ["CanonicalSmiles", "Selectivity"]
    if not (os.path.exists('dfs.p') and os.path.exists('res_df.p')):
        dfs = []
        well_data_df = None
        for k in dict_path.keys():
            for s in dict_path[k]["sheet"]:
                if s == dict_path["qHTS"]["sheet"][-1]:
                    well_data_df = pd.read_excel(dict_path[k]["path"], sheet_name=s)
                else:
                    dfs.append((s, pd.read_excel(dict_path[k]["path"], sheet_name=s)))
    
        # Remove rows with no STF_ID , uM, %Pos  W2/Med and %Inh Cells/High
        well_data_df = well_data_df.dropna(subset=ft)
        for f in ft:
            well_data_df = well_data_df[well_data_df[f].astype(bool)]
        
        pickle.dump(dfs, open("dfs.p", "wb"))
        pickle.dump(well_data_df, open("res_df.p", "wb"))
    else:
        dfs = pickle.load(open("dfs.p", "rb"))
        well_data_df = pickle.load(open("res_df.p", "rb"))

    final_df = pd.DataFrame(columns=ft+additional_ft+["target"])

    print("Building " + path)
    nihcc = 0
    not_found = 0
    found = 0
    for i, row in well_data_df.iterrows():
        added = False

        # Get STF_ID in the row
        id = row[ft[0]].strip()

        # Looking for SMILE string into the other dataframes
        for name, df in dfs:

            # Looking for the row with same id
            # NOTE: can return multiple matches
            if "NIHCC" in name:
                label = "STF_ID"
            else:
                label = "Corp_ID"
            r = df[df[label] == id]

            if len(r) > 0:
                for ii, el in r.iterrows():

                    final_row = {ft[0]: id,
                                 ft[1]: row[ft[1]],
                                 ft[2]: row[ft[2]],
                                 ft[3]: row[ft[3]],
                                 "target": -1 if row[ft[2]] <= 70 else 1 if row[ft[2]] > 130 else 0}

                    for aft in additional_ft:
                        if not pd.isnull(el[aft]):
                            if isinstance(el[aft], str) and (not el[aft].strip() == "") and ("no data" not in el[aft].lower().strip()):
                                final_row[aft] = el[aft].strip()
                            elif isinstance(el[aft], float):
                                final_row[aft] = el[aft]
                            elif isinstance(el[aft], int):
                                final_row[aft] = el[aft]

                    if len(final_row.keys()) == len(ft) + len(additional_ft) + 1:
                        final_df = final_df.append(final_row, ignore_index=True)
                        if "NIHCC" in name:
                            nihcc += 1
                        found += 1
                        added = True
                        break

                if added:
                    break

        if not added:
            # Not added for missing values or not found
            # print(row[ft[0]], " not found or excluded for missing values")
            not_found += 1

    if found > 0:
        print(found, " elements found")
    if not_found > 0:
        print(not_found, " elements not found or excluded for missing values")
    if nihcc == 0:
        print("no from NIHCC")

    # Drop duplicate
    final_df.drop_duplicates(inplace=True)

    # Some compound are tested twice, if the target is different for at least one dose then discard the compound
    # otherwise keep the first
    dd = None
    doses = final_df["uM"].unique()
    for s in final_df["CanonicalSmiles"].unique():
        s_comp = final_df[final_df["CanonicalSmiles"] == s]
        diff = False
        if len(s_comp["%Pos  W2/Med"]) > 5:
            for dose in doses:
                if len(final_df[(final_df["CanonicalSmiles"] == s) & (final_df["uM"] == dose)]["target"].unique()) > 1:
                    diff = True
                    break
        if not diff:
            if dd is None:
                dd = s_comp.drop_duplicates(["CanonicalSmiles", "uM", "target"])
            else:
                dd = dd.append(s_comp.drop_duplicates(["CanonicalSmiles", "uM", "target"]), ignore_index=True)
    final_df = dd

    final_df.to_csv(path, index=False)

    n_doses = len(final_df["uM"].drop_duplicates())
    print("dataset length:", len(final_df), "\nnumber of doses:", n_doses, "\nunique compound number:", len(final_df)/n_doses,"\nReal unique compound number:", len(final_df["CanonicalSmiles"].unique()))


def one_of_k_encoding(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_extract_NGFP(mol, idx):
    a_ft = []
    for atom in mol.GetAtoms():
        ft =  one_of_k_encoding(atom.GetSymbol(),
                             ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Unknown']) \
           + one_of_k_encoding(atom.GetDegree(), list(range(7))) \
           + one_of_k_encoding(atom.GetTotalNumHs(), list(range(6))) \
           + one_of_k_encoding(atom.GetImplicitValence(), list(range(7))) \
           + one_of_k_encoding(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                SP3D, Chem.rdchem.HybridizationType.SP3D2]) \
           + [atom.GetIsAromatic()]

        a_ft.append(ft)
    return pd.DataFrame(a_ft, index=[idx] * len(a_ft))


def atom_extract(mol, idx):
    a_ft = []
    for atom in mol.GetAtoms():
        a_ft.append([atom.GetAtomicNum(),
                        atom.GetFormalCharge(),
                        atom.GetImplicitValence(),
                        atom.GetExplicitValence(),
                        atom.GetMass(),
                        atom.GetNumExplicitHs(),
                        atom.GetNumImplicitHs(),
                        atom.GetNumRadicalElectrons(),
                        int(atom.IsInRing()),
                        int(atom.GetIsAromatic())])

        h = atom.GetHybridization()
        hyb = [0] * 8
        hyb[int(h)] += 1
        a_ft[-1] += list(map(lambda s: int(h == s), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]))

        allowable_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Na',  'As', 'I',
                           'B', 'Zn', 'Li', 'Au', 'Pt', 'Hg', 'Unknown']
        s = atom.GetSymbol()
        if s not in allowable_atoms:
            s = allowable_atoms[-1]
        a_ft[-1] += list(map(lambda x: int(x == s), allowable_atoms))

    return pd.DataFrame(a_ft, index=[idx] * len(a_ft))


def bond_extract(mol, idx):
    b_ft = []
    for bond in mol.GetBonds():
        b_ft.append([int(bond.GetIsConjugated()),
                     int(bond.IsInRing())])

        btype = [0 for _ in range(4)]
        s = bond.GetBondType()
        btype[int(s) - 1 if int(s) <= 3 else 3] = 1
        b_ft[-1] += btype

    return pd.DataFrame(b_ft, index=[idx] * len(b_ft))


def create_atom_bond_df(df, atom_ft_extractor, bond_ft_extractor=None, atom_scaler=None, bond_scaler=None):

    atom_df = pd.DataFrame()
    bond_df = pd.DataFrame()

    smile = "CanonicalSmiles"

    for i, row in df.iterrows():
        # Extract molecule from smiles representation
        m = Chem.MolFromSmiles(row[smile])

        if row[smile] not in atom_df.index:
            # Extract attributes for atoms in the molecules
            atom_df = atom_df.append(atom_ft_extractor(m, row[smile]))

        if bond_ft_extractor is not None:
            if row[smile] not in bond_df.index:
                # Extract attributes for bonds in the molecules
                bond_df = bond_ft_extractor(m, row[smile]) if i == 0 else bond_df.append(bond_ft_extractor(m, row[smile]))

    # Scale attributes
    if atom_scaler is not None:
        if atom_scaler.mean is not None:
            atom_df = atom_scaler.transform(atom_df)
        else:
            atom_df = atom_scaler.fit_transform(atom_df)

    if bond_scaler is not None:
        if bond_scaler.mean is not None:
            bond_df = bond_scaler.transform(bond_df)
        else:
            bond_df = bond_scaler.fit_transform(bond_df)

    return atom_df, bond_df, atom_scaler, bond_scaler


def create_train_test_df(path="./Datasets/torch_dataset.csv", source="./Datasets/", test_size=0.2, dose=True, NGFP= False, random_seed=None):
    df = pd.read_csv(path)

    if not dose:
        df.drop("uM", axis=1, inplace=True)
        tmp = pd.DataFrame()
        for smile in df['CanonicalSmiles'].unique():
            # Use as target the class with more occurrences
            cmps = df[df['CanonicalSmiles'] == smile]
            t = cmps["target"].value_counts().idxmax()
            tmp = tmp.append(cmps[cmps['target'] == t].iloc[[-1]])
        df = tmp

    ft = df.columns.tolist()
    ft.remove("target")
    x_train, x_test, y_train, y_test = train_test_split(df[ft], df["target"], test_size=test_size, random_state=random_seed, shuffle=True, stratify=df["target"])

    # Create training set
    train_df = pd.DataFrame(columns=df.columns)
    train_df[ft] = x_train
    train_df["target"] = y_train

    if dose:
        # Scale column uM
        train_scaler = Scaler(columns=["uM"])
        train_df[ft] = train_scaler.fit_transform(train_df[ft])

    train_df.to_csv(source+"train_dataset.csv", index=False)

    # Create atom-bond training sets
    if NGFP:
        atom_df, bond_df, atom_scaler, bond_scaler = create_atom_bond_df(train_df, atom_extract_NGFP, bond_extract)
    else:
        atom_df, bond_df, atom_scaler, bond_scaler = create_atom_bond_df(train_df, atom_extract, bond_extract, Scaler(columns=[i for i in range(8)]), None)
    atom_df.to_csv(source + "train_atom.csv")
    if bond_df is not None:
        bond_df.to_csv(source + "train_bond.csv")

    # Creating test set
    test_df = pd.DataFrame(columns=df.columns)
    test_df[ft] = x_test
    test_df["target"] = y_test

    # what we want:
    #    scaled_train =  (train - train_mean) / train_std_deviation
    #    scaled_test = (test - train_mean) / train_std_deviation

    if dose:
        # Scale columns uM based on train_scaler
        test_df[ft] = train_scaler.transform(test_df[ft])
    test_df.to_csv(source+"test_dataset.csv", index=False)
    if NGFP:
        atom_df, bond_df, atom_scaler, bond_scaler = create_atom_bond_df(test_df, atom_extract_NGFP, bond_extract)
    else:
        atom_df, bond_df, atom_scaler, bond_scaler = create_atom_bond_df(test_df, atom_extract, bond_extract, atom_scaler, bond_scaler)
    atom_df.to_csv(source + "test_atom.csv")
    if bond_df is not None:
        bond_df.to_csv(source + "test_bond.csv")


class Scaler:
    def __init__(self, columns, mean=None, std=None):
        self.mean = mean
        self.std = std
        self.columns = columns

    def fit(self, data):
        if self.columns is not None:
            data = data[self.columns]
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        print(self.columns)
        if self.mean is None or self.std is None:
            raise NotFittedError("This instance of " + self.__class__.__name__ + " is not fitted yet")

        for c in self.columns:
            data.loc[:, c] = (data.loc[:, c] - self.mean[c]) / self.std[c] if self.std[c] > 0 else 0

        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class NotFittedError(Exception):
    def __init__(self, message):
        super().__init__(message)


def create_static_fp_dataset(complex_cv=False, original_data_path="./Datasets/train_dataset.csv", saving_path="./Datasets/train_random_forest"):
    for radius in [3,4,5]:
        for length in [128, 512, 1024, 2048, 4096]:
            ft = {
                "c": False, # useChirality 
                "b": True,  # useBondTypes
                "f": True,  # useFeatures
                "l": length,
                "r": radius
            }

            train_name = saving_path + ("_group-" if complex_cv else "-")\
                                     + ("T" if ft["c"] else "F")\
                                     + ("T" if ft["b"] else "F")\
                                     + ("T" if ft["f"] else "F")\
                                     + '-L' + str(ft["l"]) + '-R' + str(ft["r"]) + ".csv"
            test_name = train_name.replace('train', 'test')

            # Read datasets
            tr_name = original_data_path
            ts_name = original_data_path.replace('train', 'test')
            train = pd.read_csv(tr_name)
            test = pd.read_csv(ts_name)

            # Extracting Morgan Fingerprint
            y_tr = train["target"]
            x_tr = []
            for _, row in train.iterrows():
                mol = Chem.MolFromSmiles(row["CanonicalSmiles"])
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=ft["c"], useBondTypes=ft["b"], useFeatures=ft["f"], radius=ft["r"], nBits=ft["l"])
                tmp = (list(fingerprint) + [row['uM'], mol.GetNumAtoms()] if complex_cv 
                       else list(fingerprint) + [row['uM']])
                x_tr.append(tmp)

            y_ts = test["target"]
            x_ts = []
            for _, row in test.iterrows():
                mol = Chem.MolFromSmiles(row["CanonicalSmiles"])
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=ft["c"], useBondTypes=ft["b"], useFeatures=ft["f"], radius=ft["r"], nBits=ft["l"])
                x_ts.append(list(fingerprint) + [row['uM']])

            df = pd.DataFrame(x_tr)
            df["y"] = y_tr
            df.to_csv(train_name, index=False)

            df = pd.DataFrame(x_ts)
            df["y"] = y_ts
            df.to_csv(test_name, index=False)
                

def create_bioval_set(source, df_path):
    df = pd.read_csv(df_path)
    df2 = pd.read_csv(source + ".csv")

    ft = df.columns.tolist()
    ft.remove("target")
    x_train, x_test, y_train, y_test = train_test_split(df[ft], df["target"], test_size=0.2, random_state=42, shuffle=True, stratify=df["target"])

    train_df = pd.DataFrame(columns=df.columns)
    train_df[ft] = x_train
    train_df["target"] = y_train
    train_scaler = Scaler(columns=["uM"])
    train_df[ft] = train_scaler.fit_transform(train_df[ft])
    del x_train, y_train, x_test, y_test, train_df

    df2["uM"] = (df2.loc[:, "uM"] - train_scaler.mean["uM"]) / train_scaler.std["uM"] if train_scaler.std["uM"] > 0 else 0
    df2.to_csv(source+"_scaled.csv", index=False)

    atom_df, bond_df, atom_scaler, bond_scaler = create_atom_bond_df(df2, atom_extract_NGFP, bond_extract)
    atom_df.to_csv(source + "_atom.csv")
    if bond_df is not None:
        bond_df.to_csv(source + "_bond.csv")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bioval', help='If True build dataset for the final biological validation', action='store_true')
    parser.add_argument('--parse_SPARK', help='If True parse the original SPARK data', action='store_true')
    parser.add_argument('--source_path', help='The path of the data used for biological validation', type=str, default="./Datasets/sweetlead")
    args = parser.parse_args()

    if args.parse_SPARK:
        ## Parse SPARK data
        start = time.time()
        create_compound_df()
        t = time.time() - start
        print('%d hours, %d minutes, %d seconds, %.3d ms' % (t // 3600, t % 3600 // 60, t % 60, (t % 1)*1000))

    if args.bioval:
        ## Build dataset for the final biological validation on sweetlead
        start = time.time()
        create_bioval_set(args.source_path, "./Datasets/torch_dataset.csv")
        t = time.time() - start
        print('%d hours, %d minutes, %d seconds, %.3d ms' % (t // 3600, t % 3600 // 60, t % 60, (t % 1)*1000))

    else:
        ## Get atom-bond features and split into train-test set
        ## for dynamic fingerprint dataset
        start = time.time()
        create_train_test_df(source="./Datasets/NGFP_", dose=True, NGFP=True, random_seed=42)
        t = time.time() - start

        ## Build static fingerprint dataset
        start = time.time()
        create_static_fp_dataset(complex_cv=True, saving_path="./Datasets/static_fp_complex_cv/train_random_forest")
        create_static_fp_dataset(complex_cv=False, saving_path="./Datasets/static_fp/train_random_forest")
        t = time.time() - start
        print('%d hours, %d minutes, %d seconds, %.3d ms' % (t // 3600, t % 3600 // 60, t % 60, (t % 1)*1000))
