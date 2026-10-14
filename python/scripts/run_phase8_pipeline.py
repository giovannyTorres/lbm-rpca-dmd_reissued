#!/usr/bin/env python3
"""Run reconstruction, holdout forecasting and inspectable phase-8 outputs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['smoke','full'],default='smoke')
    parser.add_argument('--reuse-completed',action='store_true',help='Reuse only a complete result with matching configuration hash')
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    configurations=[('run_research.py','extended_smoke.json' if args.mode=='smoke' else 'extended_physics.json'),
                    ('run_forecasting.py','forecast_smoke.json' if args.mode=='smoke' else 'forecasting_campaign.json')]
    for script,name in configurations:
        config_path=root/'configs/research'/name
        configuration=json.loads(config_path.read_text())
        complete=root/'results/research'/configuration['campaign_id']/'completion.json'
        reusable=False
        if args.reuse_completed and complete.exists():
            import hashlib
            expected=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
            payload=json.loads(complete.read_text())
            provenance_path=complete.parent/'provenance.json'
            source_matches=False
            if provenance_path.exists():
                provenance=json.loads(provenance_path.read_text())
                recorded=provenance.get('source_files_sha256',{})
                source_matches=bool(recorded) and all((root/name).exists() and
                    hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in recorded.items())
            reusable=payload.get('complete') and payload.get('config_sha256')==expected and source_matches
        if not reusable:
            subprocess.run([sys.executable,str(root/'python/scripts'/script),'--config',str(config_path)],cwd=root,check=True)
        if script=='run_research.py':
            subprocess.run([sys.executable,str(root/'python/scripts/analyze_research.py'),'--campaign',configuration['campaign_id']],cwd=root,check=True)
    print('FASE 8 complete: results/research; evaluate source provenance before reusing old outputs')


if __name__=='__main__':main()
