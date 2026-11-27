"""Build all unique paper label profiles selected by the registered configs."""
import argparse
import json
from pathlib import Path
from model.experiment_config import resolve,read_json,MODEL
from model.label_profiles import ensure_store

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--paths')
    parser.add_argument('--config',default='all');args=parser.parse_args()
    models=[r['id'] for r in read_json(MODEL/'model_registry.json')] if args.config=='all' else [args.config]
    done=set();records=[]
    for name in models:
        cfg=resolve(name,args.paths)
        for valid in [False,True]:
            if cfg['experiment']['labels']['native_only']:continue
            from model.label_profiles import store_path
            path=str(store_path(cfg,valid))
            if path not in done:ensure_store(cfg,valid);done.add(path);print(path,flush=True)
            records.append(dict(model=cfg['id'],paper_name=cfg['paper_name'],stage='valid' if valid else 'train',store=path))
    destination=Path(cfg['paths']['label_store_directory'])/'model_profiles.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
