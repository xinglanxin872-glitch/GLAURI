"""Dataset construction settings, separate from model-training configuration."""
import json
from pathlib import Path
from src.data_encoding import categ_to_resnames
config_data={
    'l_types':categ_to_resnames['protein'],
    'r_types':[categ_to_resnames['protein'],categ_to_resnames['dna']+categ_to_resnames['rna'],
               categ_to_resnames['ion'],categ_to_resnames['ligand']],
    'ex_types':json.loads((Path(__file__).parent/'artifact_residue_names.json').read_text()),
}
