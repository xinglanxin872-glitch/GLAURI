"""Inference/default adapter; training resolves JSON directly without importing globals."""
from model.experiment_config import resolve
_resolved=resolve('GLAURI')
config_data=_resolved['config_data']
config_model=_resolved['config_model']
config_runtime=_resolved['experiment']['training']
