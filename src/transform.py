"""Convert the training configuration to the GCP encoder configuration."""
from omegaconf import OmegaConf
from copy import deepcopy


def build_b_cfg_from_a(config):
    gcp = deepcopy(config['gcp'])
    # ProteinWorkshop reads these compatibility attributes, then ignores/deletes
    # them. Expose only the actual nonlinearities in the editable configuration.
    module=gcp['module_cfg']
    module['scalar_nonlinearity'],module['vector_nonlinearity']=module['nonlinearities']
    gcp['layer_cfg']['mp_cfg']['self_message']=True
    return OmegaConf.create({key: gcp[key] for key in ('model_cfg', 'layer_cfg', 'module_cfg')})
