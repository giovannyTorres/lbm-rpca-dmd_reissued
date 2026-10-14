"""Execute chronological forecasting campaigns with separate fit processes."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
from .runner import assert_finite, dataset_digest, json_safe, make_dataset, write_json


def _worker(connection, dataset, noisy, configuration, prediction_path):
    try:
        from .forecasting import evaluate_forecasting
        from .cost import measure_call
        result, profile = measure_call(evaluate_forecasting, dataset, noisy, configuration, return_forecasts=True)
        predictions = result.pop("test_predictions")
        arrays = {"times": np.asarray(result["test_times"])}
        arrays.update({key: val for key,val in predictions.items() if val is not None})
        np.savez_compressed(prediction_path, **arrays)
        result["profile"] = profile
        assert_finite(result)
        connection.send(json_safe(result))
    except Exception as error:
        connection.send({"status": "failed", "failure": str(error), "traceback": traceback.format_exc()})
    finally:
        connection.close()


def _isolated(dataset, noisy, config, prediction_path, timeout):
    context=mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    parent,child=context.Pipe(duplex=False)
    process=context.Process(target=_worker,args=(child,dataset,noisy,config,prediction_path))
    process.start();child.close()
    try:
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if parent.poll(.2):
                result=parent.recv();process.join(5)
                return result
            if not process.is_alive():
                return {"status":"failed","failure":f"worker exit {process.exitcode}"}
        return {"status":"failed","failure":f"forecast timeout {timeout}s"}
    finally:
        if process.is_alive():process.terminate()
        process.join();parent.close()


def run_forecasting_campaign(config, root):
    from .forecasting import make_forecasting_noise
    from .cost import environment_metadata
    root=Path(root)
    destination=root/"results"/"research"/config["campaign_id"]
    destination.mkdir(parents=True,exist_ok=True)
    for folder in ("datasets","predictions"):(destination/folder).mkdir(exist_ok=True)
    configuration_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    write_json(destination/"provenance.json",{"config":config,"config_sha256":configuration_hash,
       "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
       "source_files_sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((root/"python/src/fluid_denoise/research").glob("*.py"))},
       "environment":environment_metadata(),"cost_scope":"one complete candidate-selection/train/validation/test evaluation per isolated process; includes tracemalloc; not repeated per-estimator timing",
       "evaluation":"chronological held-out forecasting; noisy validation selects; clean test evaluates only"})
    ledger=destination/"ledger.jsonl";ledger.write_text("")
    rows=[]
    for specification in config["cases"]:
        dataset=make_dataset(specification);assert_finite(dataset)
        np.savez_compressed(destination/"datasets"/(specification['id']+"_clean.npz"),
                            **{k:v for k,v in dataset.items() if isinstance(v,np.ndarray)})
        write_json(destination/"datasets"/(specification['id']+"_metadata.json"),{
            "case":specification,"metadata":dataset['metadata'],"shape":list(dataset['velocity'].shape),
            "dataset_sha256":dataset_digest(dataset),"dx":dataset['dx'],"dy":dataset['dy']})
        for noise in config["noise_models"]:
            for seed in config["seeds"]:
                identifier=f"{specification['id']}__{noise['kind']}__s{seed}"
                noisy,noise_metadata=make_forecasting_noise(dataset,seed=seed,**noise,
                   train_fraction=config['split']['train_fraction'],validation_fraction=config['split']['validation_fraction'])
                configuration={**config,"known_noise":{"kind":noise['kind'],"intensity":noise['intensity']}}
                result=_isolated(dataset,noisy,configuration,str(destination/"predictions"/(identifier+".npz")),float(config.get('condition_timeout_seconds',240)))
                result.update(id=identifier,case=specification['id'],noise=noise_metadata,seed=seed)
                with ledger.open('a') as handle:handle.write(json.dumps(json_safe(result),allow_nan=False,ensure_ascii=False)+'\n')
                row={"id":identifier,"case":specification['id'],"noise":noise['kind'],"seed":seed,"status":result['status']}
                if 'selection' in result:
                    row['selected_model']=result['selection']['selected_id']
                    row['candidate_count']=len(result['selection']['candidates'])
                    for prefix,key in [('selected','selected_test_metrics'),('persistence','persistence_test_metrics')]:
                        metrics=result.get(key) or {}
                        for metric in ('velocity_relative_l2','vorticity_relative_l2','extra_divergence_normalized','kinetic_energy_relative_l2','extra_vorticity_equation_residual_normalized'):
                            row[prefix+'_'+metric]=metrics.get(metric)
                    for metric in ('wall_seconds','cpu_seconds','rss_peak_bytes'):
                        row['selection_'+metric]=result['profile'][metric]
                    selected=row.get('selected_velocity_relative_l2');persistent=row.get('persistence_velocity_relative_l2')
                    row['beats_raw_persistence_velocity']=selected is not None and persistent is not None and selected<persistent
                else:row['failure']=result.get('failure','selected test forecast failed')
                rows.append(row)
                print(f"Forecast {len(rows)}: {identifier} {row.get('selected_model','failed')}",flush=True)
    fields=sorted({k for row in rows for k in row})
    with (destination/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(json_safe(rows))
    write_json(destination/'summary.json',rows)
    expected=len(config['cases'])*len(config['noise_models'])*len(config['seeds'])
    write_json(destination/'completion.json',{"conditions":len(rows),"expected_conditions":expected,"complete":len(rows)==expected,
         "failed_conditions":sum(row['status']!='ok' for row in rows),"beats_raw_persistence_velocity":sum(row.get('beats_raw_persistence_velocity',False) for row in rows),
         "evaluation":"chronological held-out forecasting","config_sha256":configuration_hash})
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/research/forecasting_campaign.json')
    args=parser.parse_args();root=Path(__file__).resolve().parents[4]
    config=json.loads((root/args.config).read_text())
    print(run_forecasting_campaign(config,root))
