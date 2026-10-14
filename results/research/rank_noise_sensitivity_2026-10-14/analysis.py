#!/usr/bin/env python3
"""Audit fixed sensitivity, expose comparisons, and render publication figures.

No fit or clean-reference model selection is performed. The fixed experimental
driver stays byte-for-byte frozen; this companion changes presentation only.
Run again after the paired benchmark finishes to verify all 24 duplicate fits.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
for _variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
                  'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_variable]='1'
import numpy as np

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[2]
sys.path.insert(0, str(ROOT/'python/scripts'))
import run_rank_noise_sensitivity as sensitivity
import run_model_fidelity_benchmark as paired


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-main-overlap',action='store_true')
    args=parser.parse_args()
    provenance=json.loads((DEST/'provenance.json').read_text())
    config=provenance['configuration']
    records=[json.loads(line) for line in (DEST/'ledger.jsonl').read_text().splitlines()]
    completion=json.loads((DEST/'completion.json').read_text())
    if not completion['complete'] or len(records)!=360:raise ValueError('Sensitivity campaign is incomplete')
    current_sources={str(path.relative_to(ROOT)):paired.sha256(path)
        for path in (ROOT/'python/src/fluid_denoise/research').glob('*.py')}
    if current_sources!=provenance['source_files_sha256']:raise ValueError('Scientific source hash mismatch')
    if paired.sha256(sensitivity.__file__)!=provenance['script_sha256']:raise ValueError('Fixed sensitivity driver changed')
    for record in records:
        target=record['condition']['intensity_relative_rms']
        if abs(record['exact_relative_noise_rms']-target)>2e-15:raise ValueError('RMS injection mismatch')
        if record['status']=='ok':
            values={row['postprocessing']:row for row in record['evaluations']}
            raw=values['raw']['metrics'];projected=values['helmholtz']['metrics']
            if abs(raw['vorticity_relative_l2']-projected['vorticity_relative_l2'])>1e-10:
                raise ValueError('Geometry-only Helmholtz changed discrete curl')
    identities={record['condition_fingerprint'] for record in records}
    if len(identities)!=360:raise ValueError('Duplicate sensitivity condition fingerprints')
    observed_conditions={}
    for record in records:
        condition=record['condition']
        key=tuple(condition[name] for name in ('flow','noise','intensity_relative_rms','seed'))
        observed_conditions.setdefault(key,condition['observed_sha256'])
        if observed_conditions[key]!=condition['observed_sha256']:raise ValueError('Unpaired observed matrix')
    for flow in config['flows']:
        dataset,source=paired.load_flow(flow)
        if source['sha256']!=provenance['source_datasets'][flow]['sha256']:raise ValueError('Input archive changed')
        clean_rms=float(np.sqrt(np.mean(dataset['velocity']**2)))
        for key,expected in observed_conditions.items():
            name,noise,intensity,seed=key
            if name!=flow:continue
            perturbation=paired.observation_noise(dataset['velocity'].shape,dataset['times'],noise,
                seed,clean_rms,dict(config,intensity_relative_rms=intensity))
            observed=dataset['velocity']+perturbation
            if sensitivity.array_sha(observed)!=expected:raise ValueError('Reproduced noise observation hash mismatch')
    rows=json.loads((DEST/'summary.json').read_text())
    groups=json.loads((DEST/'aggregates.json').read_text())
    oracle=json.loads((DEST/'oracle_floor.json').read_text())
    if len(rows)!=1080 or len(groups)!=360:raise ValueError('Fit/evaluation/aggregate counts mismatch')
    cost_rows=[]
    for record in records:
        condition=record['condition'];profile=record.get('fit_profile',{});diagnostics=record.get('diagnostics',{})
        accepted={value['postprocessing']:value['accepted'] for value in record['evaluations']}
        cost_rows.append({key:condition[key] for key in
            ('flow','noise','intensity_relative_rms','seed','rank','method')}
            | dict(status=record['status'],effective_rank=diagnostics.get('effective_rank'),
                svd_calls=diagnostics.get('svd_calls'),flops_proxy=diagnostics.get('flops_proxy'),
                flops_proxy_definition=diagnostics.get('flops_proxy_definition'))
            | {key:profile.get(key) for key in ('cpu_seconds','wall_seconds','rss_peak_bytes',
                'rss_peak_before_bytes','python_traced_peak_bytes')}
            | {f'accepted_{variant}':accepted.get(variant,False) for variant in config['postprocessing']})
    fieldnames=list(cost_rows[0])
    with (DEST/'rank_cost_diagnostics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fieldnames);writer.writeheader();writer.writerows(cost_rows)
    summaries=[]
    for flow in config['flows']:
        for noise in config['noise_models']:
            for intensity in config['intensities_relative_rms']:
                for method in config['methods']:
                    entry=dict(flow=flow,noise=noise,intensity_relative_rms=intensity,
                               method_family=paired.METHODS[method],ranks={})
                    for rank in config['ranks']:
                        entry['ranks'][str(rank)]={}
                        for post in config['postprocessing']:
                            matching=[group for group in groups if
                                (group['flow'],group['noise'],group['intensity_relative_rms'],group['method'],
                                 group['rank'],group['postprocessing'])==(flow,noise,intensity,method,rank,post)]
                            if len(matching)!=1:raise ValueError('Aggregate lookup is not unique')
                            group=matching[0]
                            entry['ranks'][str(rank)][post]={key:group[key] for key in
                                ('accepted_count','sample_count',*sensitivity.METRICS)}
                    summaries.append(entry)
    overlap_records=[record for record in records if record['overlap_model_benchmark']]
    main_path=ROOT/'results/research/model_fidelity_benchmark_2026-10-14/ledger.jsonl'
    main_provenance=ROOT/'results/research/model_fidelity_benchmark_2026-10-14/provenance.json'
    main_records=[json.loads(line) for line in main_path.read_text().splitlines()] if main_path.exists() else []
    main_config=json.loads(main_provenance.read_text())['configuration'] if main_provenance.exists() else None
    overlaps=[]
    for record in overlap_records:
        condition=record['condition']
        match=[row for row in main_records if (row['flow'],row['noise'],row['seed'],row['method'])==
            tuple(condition[key] for key in ('flow','noise','seed','method'))]
        entry={key:condition[key] for key in ('flow','noise','seed','method','observed_sha256')}
        entry['present_in_main']=bool(match)
        if match:
            if len(match)!=1:raise ValueError('Main duplicate lookup is not unique')
            other=match[0]
            if other['noise_metadata']['observation_sha256']!=condition['observed_sha256']:
                raise ValueError('Nominal duplicate has different observed matrix')
            params=paired.method_parameters(condition['method'],condition['seed'],main_config)
            if params!=condition['algorithm_parameters'] or main_config['rank']!=condition['rank']:
                raise ValueError('Duplicate estimator parameters differ')
            entry.update(observed_hash_equal=True,algorithm_parameters_equal=True,
                main_condition_fingerprint=other['fingerprint'],main_status=other['status'])
            if other['status']=='ok' and record['status']=='ok':
                main_eval={value['variant']:value for value in other['evaluations']}
                maximum=0.
                for value in record['evaluations']:
                    for metric in sensitivity.METRICS:
                        actual=value['metrics'][metric]; expected=main_eval[value['postprocessing']]['metrics'][metric]
                        if actual is not None and expected is not None:maximum=max(maximum,abs(actual-expected))
                if maximum>2e-10:raise ValueError('Identical observations/parameters returned different deterministic fidelity')
                entry['maximum_abs_fidelity_metric_difference']=maximum
        overlaps.append(entry)
    verified=sum(row['present_in_main'] for row in overlaps)
    if args.require_main_overlap and verified!=24:raise ValueError(f'Only {verified}/24 overlaps are available')
    # Use the frozen renderer. This override corrects editorial spelling only.
    from matplotlib.figure import Figure
    original_title=Figure.suptitle
    def publication_title(figure,title,*args,**kwargs):
        title=title.replace('incomprensibilidad','incompresibilidad')
        title=title.replace('compresibilidad y sesgo del prior','compresión matricial y sesgo del prior')
        title=title.replace('mediana y rango de tres semillas','mediana y mínimo–máximo de tres semillas')
        title=title.replace('mediana/rango de tres semillas','mediana y mínimo–máximo de tres semillas')
        return original_title(figure,title,*args,**kwargs)
    Figure.suptitle=publication_title
    sensitivity.figures(groups,oracle,config)
    selected=[]
    for path in sorted((DEST/'reconstructions').glob('*.npz')):
        with np.load(path) as arrays:
            selected.append(dict(path=str(path.relative_to(ROOT)),sha256=paired.sha256(path),
                bytes=path.stat().st_size,velocity_shape=list(arrays['velocity'].shape),
                times=arrays['times'].tolist(),finite=bool(np.isfinite(arrays['velocity']).all()),
                interpretation='Selected seed101/rank8/RMS20% H+k10 full-window estimate; publication of NPZ is optional.'))
    def percentage(value):
        if value is None:return 'no disponible'
        return f'{100*value["median"]:.2f} [{100*value["minimum"]:.2f}, {100*value["maximum"]:.2f}]'
    prose=[
        '# Sensibilidad de rango, intensidad y fidelidad física',
        '',
        'Esta campaña contrasta POD y TLS-DMD para comprobar cuánto dependen las conclusiones de fijar un único rango y una única intensidad de ruido. El diseño se fijó antes de ejecutar sus resultados: **360 ajustes**, sobre **36 observaciones diferentes**, y **1080 evaluaciones** de los mismos ajustes con tres tratamientos espaciales. Las tres semillas son nuevas realizaciones de ruido sobre dos trayectorias ya estudiadas; no prueban generalización a flujos nuevos.',
        '',
        'Los campos son Navier–Stokes multiescala (128², T=4, 41 estados) y Oldroyd-B (128², T=2.4, 25 estados). No se agrupan como una comparación física común de materiales: tienen trayectorias y horizontes distintos. Se compara el comportamiento de los reconstruidores dentro de cada campo.',
        '',
        'Se usan ruido gaussiano IID y ruido impulsivo sobre el 1% de las **entradas de la matriz**, normalizados a RMS global 5%, 20% y 40% de la señal limpia; rangos 2, 4, 8, 12 y 16; semillas 101, 307 y 911. La amplitud sintética usa la referencia limpia para definir el experimento, pero el estimador recibe exclusivamente observación y tiempos. Igual RMS no implica igual dificultad entre distribuciones.',
        '',
        'Cada estimación se evalúa sin tratamiento (Raw), con proyección de Helmholtz (H) y con H más corte isotrópico físico k≤10. El corte 10 se hereda de una exploración previa como supuesto de suavidad; no es un parámetro elegido mediante el mínimo de error limpio de esta campaña. Los alias de implementación `pod_r8` y `tls_r8` identifican las familias del benchmark; el rango efectivo de esta campaña lo determina la columna `rank`.',
        '',
        '## Cómo leer los resultados',
        '',
        'Los errores se calculan sobre el campo completo y toda la ventana temporal, incluyendo todos los modos limpios. Cada celda muestra **mediana [mínimo, máximo] de tres semillas**, en porcentaje. No es un intervalo de confianza. La aceptación exige convergencia y los cuatro límites declarados: error de velocidad ≤15%, vorticidad ≤25%, divergencia extra normalizada ≤10% y energía cinética ≤15%. Cumplirlos es una tolerancia de reconstrucción, no certificación DNS ni solución exacta de las ecuaciones.',
        '',
        f'Completitud: {completion["fits"]} ajustes; {completion["failed_fits"]} fallos; {completion["nonconverged_fits"]} ajustes sin convergencia. Evaluaciones aceptadas: Raw {completion["accepted_by_postprocessing"]["raw"]}/360; H {completion["accepted_by_postprocessing"]["helmholtz"]}/360; H+k≤10 {completion["accepted_by_postprocessing"]["helmholtz_k10"]}/360. Los tres tratamientos reutilizan cada ajuste y no constituyen tres experimentos independientes.',
        '',
        'La tabla siguiente muestra el contraste fijado de RMS 20% para rangos 2, 8 y 16. El resto de rangos e intensidades permanece en las tablas completas. H conserva exactamente la vorticidad discreta: su beneficio de divergencia o velocidad no se interpreta como recuperación de gradientes.',
        ''
    ]
    for flow in config['flows']:
        for noise in config['noise_models']:
            prose += [f'### {sensitivity.FLOW_LABELS[flow]} — {sensitivity.NOISE_LABELS[noise]}','',
                '| Modelo | Rango | Error ω Raw [%] | Error ω H+k≤10 [%] | Error u H+k≤10 [%] | Aceptadas H+k≤10 |',
                '|---|---:|---:|---:|---:|---:|']
            for method in config['methods']:
                for rank in (2,8,16):
                    values={group['postprocessing']:group for group in groups if
                        (group['flow'],group['noise'],group['intensity_relative_rms'],group['method'],group['rank'])
                        ==(flow,noise,.2,method,rank)}
                    raw=values['raw'];band=values['helmholtz_k10']
                    prose.append(f'| {sensitivity.METHOD_LABELS[method]} | {rank} | {percentage(raw["vorticity_relative_l2"])} | {percentage(band["vorticity_relative_l2"])} | {percentage(band["velocity_relative_l2"])} | {band["accepted_count"]}/3 |')
            prose.append('')
    prose += ['## Compresión matricial de la señal limpia y sesgo del filtro','',
        'El oráculo POD se calcula **por separado** sobre la matriz limpia para diagnosticar cuánto puede aproximarse con rango reducido. Esta compresión matricial no describe la compresibilidad del fluido. Sólo su error de velocidad en norma Frobenius es una cota inferior entre matrices de rango r. Su error de vorticidad es descriptivo: el SVD no minimiza el error de derivadas. El oráculo no entra en los estimadores ruidosos ni selecciona un rango.','',
        '| Campo | Rango | Cota limpia de error u [%] | Error ω de POD limpio [%] |',
        '|---|---:|---:|---:|']
    for row in oracle:
        prose.append(f'| {sensitivity.FLOW_LABELS[row["flow"]]} | {row["rank"]} | {100*row["velocity_rank_frobenius_lower_bound"]:.5f} | {100*row["vorticity_clean_pod_diagnostic"]:.5f} |')
    prose+=['']
    for flow in config['flows']:
        row=next(row for row in oracle if row['flow']==flow)
        prose.append(f'Aplicar el corte k≤10 directamente a la referencia limpia de {sensitivity.FLOW_LABELS[flow]} produce {100*row["clean_band_velocity_error"]:.3f}% de error de velocidad y **{100*row["clean_band_vorticity_error"]:.3f}% de error de vorticidad**. Por tanto, la mejora al filtrar ruido tiene un coste físico explícito y dependiente del campo.')
        prose.append('')
    prose+=['## Figuras y datos','',
        '- [Velocidad frente a rango e intensidad](figures/rank_velocity.pdf).',
        '- [Vorticidad frente a rango e intensidad](figures/rank_vorticity.pdf).',
        '- [Separación de algoritmo, proyección y corte](figures/rank_postprocessing.pdf).',
        '- [Aceptación de cada rango/intensidad](figures/acceptance_rank_noise.pdf).',
        '- [Oráculo limpio y sesgo del prior](figures/clean_oracle_floor.pdf).','',
        'Las fuentes inspectables son `ledger.jsonl` (cada ajuste, diagnósticos, perfiles y métricas completas), `summary.csv` (1080 evaluaciones), `aggregates.json` (mediana/mínimo/máximo), `oracle_floor.csv` (diagnóstico limpio), `diagnostic_comparisons.json` y `rank_cost_diagnostics.csv` (360 filas de coste por ajuste real). `analysis_verification.json` verifica hashes científicos y de entrada, las 36 observaciones regeneradas y la conservación de vorticidad bajo H.','',
        'La medida de coste separa ajuste y postprocesado de la evaluación. CPU, tiempo de pared, RSS absoluto y memoria trazada se registran con un proceso nuevo y un hilo BLAS por ajuste. El RSS de un postprocesado conserva el pico histórico de ese proceso; no es memoria incremental del filtro. Los proxies aritméticos no son FLOPs medidos. Tres semillas son observaciones diferentes, no tres repeticiones de coste sobre una misma matriz; los tiempos de un equipo compartido son descriptivos.','',
        f'Los 24 ajustes de rango 8/RMS 20% coinciden con condiciones del benchmark principal si se verifican observación SHA256 y parámetros. Estado de esa comprobación: **{verified}/24** disponibles y verificados en `overlap_verification.json`. Se conservan como análisis de sensibilidad, sin sumarlos como evidencia independiente.','',
        '## Reproducción','',
        'Desde la raíz del repositorio, con las dependencias instaladas y los dos archivos completos de simulación generados por las campañas de mallas ampliadas:','',
        '```bash',
        'python python/scripts/run_rank_noise_sensitivity.py --resume',
        'python results/research/rank_noise_sensitivity_2026-10-14/analysis.py',
        '# Cuando también termine el benchmark principal:',
        'python results/research/rank_noise_sensitivity_2026-10-14/analysis.py --require-main-overlap',
        '```','',
        'La caché local valida configuración, fuentes, entradas y condición; no se publica por duplicar el ledger. Los NPZ de reconstrucción se guardan sólo para semilla 101, rango 8 y RMS 20%, con H+k≤10, y figuran en `selected_reconstructions_manifest.json`. Los archivos de muestras visuales de cinco estados del repositorio no sustituyen las historias completas necesarias para recalcular las métricas. `provenance.json` conserva los contratos de los helpers y sus hashes al inicio: ajustes posteriores de lectura de otros materiales no se presentan como un módulo idéntico.']
    (DEST/'README.md').write_text('\n'.join(prose)+'\n')
    paired.write_json(DEST/'diagnostic_comparisons.json',dict(
        quality_thresholds=config['quality_thresholds'],rank_grid_fixed=True,
        no_rank_selection_from_clean=True,no_independent_new_flow_validation=True,
        comparisons=summaries,clean_diagnostics=oracle,
        semantics='Each median and minimum/maximum describe exactly three noise seeds on one fixed trajectory. Only velocity oracle POD gives a rank-constrained Frobenius lower bound. Vorticity oracle is descriptive.',
        comparisons_never_sum_overlap_as_independent=True))
    paired.write_json(DEST/'rank_cost_scope.json',dict(rows=360,one_row_per_actual_fit=True,
        measured_fit_cpu_seconds_sum=sum(row.get('fit_profile',{}).get('cpu_seconds',0.) for row in records),
        measured_postprocess_cpu_seconds_sum=sum(value['postprocessing_profile']['cpu_seconds'] for row in records for value in row['evaluations']),
        cpu_totals_scope='Sum of measured model/postprocess calls, excluding evaluation, noise generation, oracle diagnostics, plotting, process startup and I/O. This is not total campaign CPU or electricity.',
        separate_from_1080_postprocessing_evaluations=True,measurement='One fresh single-BLAS-thread child per condition. Fit CPU, wall, absolute lifetime RSS and Python-traced allocations; evaluation excluded.',
        arithmetic_proxy='Diagnostics count algorithm work with documented omitted operations. It is a dominant-arithmetic proxy, not a measured exact FLOP census.',
        postprocessing='Each fit has three separately measured postprocessing stages in ledger; peak RSS of a later stage includes the historical child-process peak and must not be treated as incremental postprocessing memory.',
        postprocessing_fft_diagnostic='The inherited fft_calls=2 describes the Helmholtz projection substage only. H+k10 then adds another FFT and inverse FFT: four transform calls in the combined pipeline, not two. Postprocess CPU/wall profiles measure the full combined function.',
        uncertainty='Three seeds are different observed matrices, not repeated-cost measurements of the same matrix. Shared-host wall timings are descriptive; no precise speed ranking or confidence interval is claimed.'))
    paired.write_json(DEST/'overlap_verification.json',dict(expected_overlapping_fits=24,
        available_verified_fits=verified,complete=verified==24,records=overlaps,
        interpretation='Same observed array plus estimator parameters; these fits duplicate the main model benchmark and are not independent evidence.',
        main_ledger_sha256=paired.sha256(main_path) if main_path.exists() else None))
    paired.write_json(DEST/'selected_reconstructions_manifest.json',selected)
    paired.write_json(DEST/'analysis_verification.json',dict(
        analysis_script_sha256=paired.sha256(__file__),fixed_driver_sha256=paired.sha256(sensitivity.__file__),
        source_hashes_unchanged=True,input_hashes_unchanged=True,unique_condition_fingerprints=360,
        regenerated_observation_hashes_match=True,unique_observations=len(observed_conditions),
        rms_injection_verified=True,helmholtz_discrete_curl_preservation_verified=True,
        ledger_sha256=paired.sha256(DEST/'ledger.jsonl'),oracle_sha256=paired.sha256(DEST/'oracle_floor.json'),
        fits=360,evaluations=1080,aggregates=360,main_overlap_verified=verified,
        figure_paths=[str(path.relative_to(ROOT)) for path in sorted((DEST/'figures').glob('*.pdf'))],
        presentation_changes_only='Frozen experiment renderer reused; spelling and minimum–maximum wording corrected in titles.'))
    print(json.dumps(dict(fits=360,evaluations=1080,unique_observations=len(observed_conditions),main_overlap_verified=verified)))


if __name__=='__main__':main()
