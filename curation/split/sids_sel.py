import numpy as np
import h5py
from data_handler import Dataset
from config import config_data
from src.dataset import select_by_interface_types
import pickle

from datetime import datetime
from src.data_encoding import categ_to_resnames
from pathlib import Path

split_dir = Path("/home/lanxin/data/Interface_prediction/processing/data/datasets")
split_prefix = "refine90_projected_split"
split_files = {
    "train": split_dir / f"{split_prefix}_train_set.txt",
    "val": split_dir / f"{split_prefix}_val_set.txt",
    "test": split_dir / f"{split_prefix}_test_set.txt",
}

config_data = {
    'dataset_filepath': "/home/lanxin/data/Interface_prediction/bio_251210.h5",
    'max_ba': 1,
    'max_size': 8192,
    'min_num_res': 48,
    'l_types': categ_to_resnames['protein'],
    'seq_dict_path':'/home/zwb/xlx/Interface_prediction/cluster/all_seq_dict.pkl',
    'r_types': [
        categ_to_resnames['protein'],
        categ_to_resnames['dna']+categ_to_resnames['rna'],
        categ_to_resnames['ion'],
        categ_to_resnames['ligand']
    ],
    'ex_types':[ 'ACE','HEX', 'TMA', 'SOH', 'P25', 
                'CCN', 'PR', 'PTN', 'NO3', 'TCN', 'BU1',
                'BCN', 'CB3', 'HCS', 'NBN', 'SO2', 'MO6',
                'MOH', 'CAC', 'MLT', 'KR', '6PH', 'MOS', 
                'UNL', 'MO3', 'SR', 'CD3', 'PB', 'ACM', 'LUT', 
                'PMS', 'OF3', 'SCN', 'DHB', 'E4N', '13P', '3PG'
                , 'CYC', 'NC', 'BEN', 'NAO', 'PHQ', 'EPE', 'BME', 
                'TB', 'ETE', 'EU', 'OES', 'EAP', 'ETX', 'BEZ', 
                '5AD', 'OC2', 'OLA', 'GD3', 'CIT', 'DVT', 'OC6', 'MW1', 'OC3', 'SRT', 'LCO', 'BNZ', 'PPV', 'STE', 'PEG', 'RU', 'PGE', 'MPO', 'B3P', 'OGA', 'IPA', 'LU', 'EDO', 'MAC', '9PE', 'IPH', 'MBN', 'C1O', '1PE', 'YF3', 'PEF', 'GD', '8PE', 'DKA', 'RB', 'YB', 'GGD', 'SE4', 'LHG', 'SMO', 'DGD', 'CMO', 'MLI', 'MW2', 'DTT', 'DOD', '7PH', 'PBM', 'AU', 'FOR', 'PSC', 'TG1', 'KAI', '1PG', 'DGA', 'IR', 'PE4', 'VO4', 'ACN', 'AG', 'MO4', 'OCL', '6UL', 'CHT', 'RHD', 'CPS', 'IR3', 'OC4', 'MTE', 'HGC', 'CR', 'PC1', 'HC4', 'TEA', 'BOG', 'PEO', 'PE5', '144', 'IUM', 'LMG', 'SQU', 'MMC', 'GOL', 'NVP', 'AU3', '3PH', 'PT4', 'PGO', 'ICT', 'OCM', 'BCR', 'PG4', 'L4P', 'OPC', 'OXM', 'SQD', 'PQ9', 'BAM', 'PI', 'PL9', 'P6G', 'IRI', '15P', 'MAE', 'MBO', 'FMT', 'L1P', 'DUD', 'PGV', 'CD1', 'P33', 'DTU', 'XAT', 'CD', 'THE', 'U1', 'NA', 'MW3', 'BHG', 'Y1', 'OCT', 'BET', 'MPD', 'HTO', 'IBM', 'D01', 'HAI', 'HED', 'CAD', 'CUZ', 'TLA', 'SO4', 'OC5', 'ETF', 'MRD', 'PT', 'PHB', 'URE', 'MLA', 'TGL', 'PLM', 'NET', 'LAC', 'AUC', 'UNX', 'GA', 'DMS', 'MO2', 'LA', 'NI', 'TE', 'THJ', 'NHE', 'HAE', 'MO1', 'DAO', '3PE', 'LMU', 'DHJ', 'FLC', 'SAL', 'GAI', 'ORO', 'HEZ', 'TAM', 'TRA', 'NEX', 'CXS', 'LCP', 'HOH', 'OCN', 'PER', 'ACY', 'MH2', 'ARS', '12P', 'L3P', 'PUT', 'IN', 'CS', 'NAW', 'SB', 'GUN', 'SX', 'CON', 'C2O', 'EMC', 'BO4', 'BNG', 'MN5', 'K', 'CYN', 'H2S', 'MH3', 'YT3', 'P22', 'KO4', '1AG', 'CE', 'IPL', 'PG6', 'MO5', 'F09', 'HO', 'AL', 'TRS', 'EOH', 'GCP', 'MSE', 'AKR', 'NCO', 'PO4', 'L2P', 'LDA', 'SIN', 'DMI', 'SM', 'DTD', 'SGM', 'DIO', 'PPI', 'DDQ', 'DPO', 'HCA', 'CO5', 'PD', 'OS', 'OH', 'NA6', 'NAG', 'W', 'ENC', 'NA5', 'LI1', 'P4C', 'GLV', 'DMF', 'ACT', 'BTB', '6PL', 'BGL', 'OF1', 'N8E', 'LMT', 'THM', 'EU3', 'PGR', 'NA2', 'FOL', '543', 'PEK', 'NSP', 'PEE', 'OCO', 'CHD', 'CO2', 'TBU', 'UMQ', 'MES', 'NH4', 'CD5', 'HTG', 'DEP', 'OC1', 'KDO', '2PE', 'PE3', 'IOD', 'NDG', 'CL', 'HG', 'F', 'XE', 'TL', 'BA', 'LI', 'BR', 'TAU', 'TCA', 'SPD', 'SPM', 'SAR', 'SUC', 'PAM', 'SPH', 'BE7', 'P4G', 'OLC', 'OLB', 'LFA', 'D10', 'D12', 'DD9', 'HP6', 'R16', 'PX4', 'TRD', 'UND', 'FTT', 'MYR', 'RG1', 'IMD', 'DMN', 'KEN', 'C14', 'UPL', 'CMJ', 'ULI', 'MYS', 'TWT', 'M2M', 'P15', 'PG0', 'PEU', 'AE3', 'TOE', 'ME2', 'PE8', '6JZ', '7PE', 'P3G', '7PG', 'PG5', '16P', 'XPE', 'PGF', 'AE4', '7E8', '7E9', 'MVC', 'TAR', 'DMR', 'LMR', 'NER', '02U', 'NGZ', 'LXB', 'A2G', 'BM3', 'NAA', 'NGA', 'LXZ', 'PX6', 'PA8', 'LPP', 'PX2', 'MYY', 'PX8', 'PD7', 'XP4', 'XPA', 'PEV', '6PE', 'PEX', 'PEH', 'PTY', 'YB2', 'PGT', 'CN3', 'AGA', 'DGG', 'CD4', 'CN6', 'CDL', 'PG8', 'MGE', 'DTV', 'L44', 'L2C', '4AG', 'B3H', '1EM', 'DDR', 'I42', 'CNS', 'PC7', 'HGP', 'PC8', 'HGX', 'LIO', 'PLD', 'PC2', 'PCF', 'MC3', 'P1O', 'PLC', 'PC6', 'HSH', 'BXC', 'HSG', 'DPG', '2DP', 'POV', 'PCW', 'GVT', 'CE9', 'CXE', 'C10', 'CE1', 'SPJ', 'SPZ', 'SPK', 'SPW', 'HT3', 'HTH', '2OP', '3NI', 'BO3', 'DET', 'D1D', 'SWE', 'SOG'],
    'features_flags':(True,False, False)
    # 'r_types': [[c] for c in categ_to_resnames['protein']],
}
def dataset_key_to_sid(key):
    parts = key.split("/")
    if len(parts) >= 3:
        pdbid = parts[1].lower()
        chain = parts[2].split(":", 1)[0]
        return f"{pdbid}_{chain}"
    return key.split(":", 1)[0].lower()


def select_by_sid(dataset, sids_sel):
    print(type(sids_sel))
    sids = (
        dataset_key_to_sid(k) for k in dataset.keys
    )

    return np.fromiter((sid in sids_sel for sid in sids), dtype=bool)


def select_has_cluster(dataset):
    has_cluster = {}
    with h5py.File(dataset.dataset_filepath, "r") as hf:
        for key in np.unique(dataset.keys):
            try:
                has_cluster[key] = "cluster" in hf[f"data/structures/{key}"]
            except KeyError:
                has_cluster[key] = False
    return np.fromiter((has_cluster[key] for key in dataset.keys), dtype=bool)


if __name__ == "__main__":
    for key in ['train','val','test']:
        print(key)
        with open(split_files[key], "r") as f:
            processed = [line.strip() for line in f if line.strip()]
        dataset = Dataset(config_data)# whether to extract the backbone

        sids_sel=set(processed)

        sel_m = select_by_sid(dataset, sids_sel) # select by sids
        sel_m &= (dataset.sizes[:,1] >= config_data['min_num_res'])
        sel_m &= (dataset.sizes[:,0] <= config_data['max_size'])
        sel_m &= select_by_interface_types(dataset, config_data['l_types'])  # select by interface type
        sel_m &= select_has_cluster(dataset)
        dataset.update_mask(sel_m)
        print(len(dataset))
        with open(split_dir / f'{split_prefix}_{key}_m.pkl','wb')as f:
            pickle.dump(sel_m,f)
            