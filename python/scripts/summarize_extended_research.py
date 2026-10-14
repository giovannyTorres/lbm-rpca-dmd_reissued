#!/usr/bin/env python3
"""Produce the consolidated thesis discussion from executed source tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
RECON='extended_physics_2026-10-14'
FORECAST='forecasting_extended_2026-10-14'
COMPARISON_ATOL=1e-10


def main():
    rdir=ROOT/'results/research'/RECON;fdir=ROOT/'results/research'/FORECAST
    reconstruction=pd.read_csv(rdir/'summary.csv')
    reconstruction['quality_pass']=reconstruction.quality_pass.astype(str).str.lower().eq('true')
    complete=json.loads((rdir/'completion.json').read_text());future_complete=json.loads((fdir/'completion.json').read_text())
    if not complete['complete'] or not future_complete['complete']:raise RuntimeError('Incomplete campaign')
    records=[json.loads(s) for s in (fdir/'ledger.jsonl').read_text().splitlines()]
    future=pd.read_csv(fdir/'summary.csv')
    filtered=[]
    for record in records:
        metrics=record.get('test_metrics',{}).get('persistence_helmholtz_dealias',{})
        selected=record.get('selected_test_metrics',{}).get('velocity_relative_l2')
        raw=record.get('persistence_test_metrics',{}).get('velocity_relative_l2')
        projected=metrics.get('velocity_relative_l2')
        if any(value is None or not np.isfinite(value) for value in (selected,raw,projected)):
            raise ValueError(f"Missing finite comparison for {record['id']}")
        filtered.append({'id':record['id'],'filtered_persistence_velocity_relative_l2':metrics.get('velocity_relative_l2'),
                         'filtered_persistence_vorticity_relative_l2':metrics.get('vorticity_relative_l2'),
                         'beats_raw_persistence_velocity_atol':raw-selected>COMPARISON_ATOL,
                         'beats_filtered_persistence_velocity':projected-selected>COMPARISON_ATOL,
                         'ties_raw_persistence_velocity_atol':abs(raw-selected)<=COMPARISON_ATOL,
                         'ties_filtered_persistence_velocity_atol':abs(projected-selected)<=COMPARISON_ATOL})
    future=future.merge(pd.DataFrame(filtered),on='id',how='left')
    future.to_csv(fdir/'comparison_with_filtered_persistence.csv',index=False)
    aggregated=future.groupby(['case','noise']).median(numeric_only=True)
    aggregated.to_csv(fdir/'medians_by_noise.csv')
    tg=reconstruction.query("case=='taylor_green_ns' and noise=='gaussian'")
    raw=tg.query("method=='tls_dmd'").velocity_relative_l2.median()
    constrained=tg.query("method=='tls_dmd_helmholtz_dealias'").velocity_relative_l2.median()
    mean_fallback=0
    for line in (rdir/'ledger.jsonl').open():
        record=json.loads(line)
        if record['method']=='fb_dmd_fallback_tls' and any(f.get('diagnostics',{}).get('fallback_used') for f in record['fits']):mean_fallback+=1
    gains=int(future.beats_filtered_persistence_velocity.sum())
    raw_gains=int(future.beats_raw_persistence_velocity_atol.sum())
    newtonian_gaussian=reconstruction.query("noise=='gaussian' and case in ['taylor_green_ns','interacting_vortices_ns','decaying_turbulence_ns']")
    accepted_raw=int(newtonian_gaussian.query("postprocess=='none'").quality_pass.sum())
    accepted_projection=int(newtonian_gaussian.query("postprocess!='none'").quality_pass.sum())
    summary={'reconstruction':complete,'forecasting':future_complete,'future_beats_filtered_persistence':gains,
             'future_beats_raw_persistence_with_tolerance':raw_gains,
             'future_ties_raw_persistence_with_tolerance':int(future.ties_raw_persistence_velocity_atol.sum()),
             'future_ties_filtered_persistence_with_tolerance':int(future.ties_filtered_persistence_velocity_atol.sum()),
             'comparison_absolute_tolerance_relative_velocity_error':COMPARISON_ATOL,
             'fb_primary_failure_conditions_resolved_with_declared_tls_fallback':mean_fallback,
             'matched_gaussian_newtonian_acceptance_raw':accepted_raw,'matched_gaussian_newtonian_acceptance_projected':accepted_projection,
             'taylor_green_gaussian_raw_tls_velocity_error':raw,'taylor_green_gaussian_projected_tls_velocity_error':constrained}
    (ROOT/'results/research/extended_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    pictures=fdir/'figures';pictures.mkdir(exist_ok=True)
    selected=aggregated.loc[(slice(None),'gaussian'),:].reset_index()
    labels=['NS vórtices' if 'vortices_ns'in name else ('Oldroyd-B vórtices' if 'oldroyd' in name and 'interacting'in name else 'FENE-P vórtices' if 'interacting'in name else 'Oldroyd-B TG' if 'oldroyd'in name else 'FENE-P TG') for name in selected['case']]
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');x=np.arange(len(selected));width=.25
    for axis,metric,title in [(axes[0],'velocity_relative_l2','Velocidad futura'),(axes[1],'vorticity_relative_l2','Vorticidad futura')]:
        axis.bar(x-width,100*selected['persistence_'+metric],width,label='Persistencia observada')
        axis.bar(x,100*selected['filtered_persistence_'+metric],width,label='Persistencia filtrada')
        axis.bar(x+width,100*selected['selected_'+metric],width,label='Seleccionado en validación')
        axis.set_xticks(x,labels,rotation=25,ha='right');axis.set_ylabel('Error relativo [%]');axis.set_title(title);axis.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Pronóstico con ruido gaussiano: medianas de tres semillas; futuro excluido del ajuste')
    fig.savefig(pictures/'holdout_comparison.png',dpi=180);fig.savefig(pictures/'holdout_comparison.pdf');plt.close(fig)
    md=['# Ampliación integrada y ejecutada en reissued','',
        f"Se ejecutaron **{complete['conditions']} condiciones y {complete['fits']} ajustes** de reconstrucción, más **{future_complete['conditions']} escenarios de pronóstico** con 29 opciones predeclaradas por escenario. "
        f"La reconstrucción registró {complete['failed_fits']} fallos numéricos, {complete['not_converged_conditions']} condiciones no convergidas y {complete['quality_pass_conditions']} condiciones que cumplen las tolerancias. "
        f"El pronóstico registró {future_complete['failed_conditions']} escenarios fallidos.",'',
        '## Lo que cambió y lo que muestran los datos','',
        f"En Taylor–Green gaussiano, TLS-DMD de rango 4 tiene error mediano de velocidad **{100*raw:.3f}%**; con proyección y corte 2/3 baja a **{100*constrained:.3f}%**. "
        f"En los tres casos newtonianos y las mismas semillas, los métodos sin proyección aportan {accepted_raw} condiciones gaussianas aceptadas y los proyectados {accepted_projection}. "
        'La banda física es un prior declarado del banco, no un filtro ajustado con el limpio. La proyección no convierte toda reconstrucción en una solución de la ecuación de movimiento.','',
        f"El dispatcher fbDMD→TLS usó su alternativa explícita en **{mean_fallback} condiciones** por ramas ambiguas o mal condicionamiento. Se conservan los diagnósticos primarios y ambos costos: no se registra como fbDMD puro exitoso.",'',
        f"El modelo seleccionado con validación ruidosa mejora el error futuro de velocidad frente a persistencia sin filtrar en **{raw_gains}/{future_complete['conditions']} escenarios**, "
        f"y frente a persistencia filtrada en **{gains}/{future_complete['conditions']}**. Esto separa el beneficio de filtrado del beneficio dinámico. "
        f"Se comparan los valores originales del ledger con tolerancia absoluta {COMPARISON_ATOL:g} en el error relativo; quedan {future_complete['conditions']-raw_gains} y {future_complete['conditions']-gains} empates, respectivamente. "
        f"El contador bruto de completion.json conserva {future_complete['beats_raw_persistence_velocity']} mejoras mediante comparación estricta: incluye diferencias de redondeo que aquí se consideran empates. "
        'Los casos de vórtices mantienen errores de vorticidad altos: mejorar velocidad no demuestra capacidad de pronóstico fiable de todas las estructuras.','',
        f"![Pronóstico futuro](../../results/research/{FORECAST}/figures/holdout_comparison.png)",'',
        '## Comparación reológica con controles comunes','',
        'Se generaron seis fluidos con el mismo estado inicial, densidad, dominio, malla y tiempos. Las viscosidades HB se igualan en una tasa de corte declarada; Oldroyd-B/FENE-P comparten constantes nominales de viscosidad. '
        'Las diferencias frente al fluido newtoniano son cambios constitutivos simulados, no errores de reconstrucción. Oldroyd-B y FENE-P son cercanos en este régimen; estos ensayos no demuestran discriminación de materiales a partir de velocidad ruidosa.','',
        '![Campos finales](../../results/research/rheology_matched_2026-10-14/final_vorticity.png)','',
        '[Animación](../../results/research/rheology_matched_2026-10-14/vortices_rheology.gif) y [tabla cuantitativa](../../results/research/rheology_matched_2026-10-14/summary.csv).','',
        '## Evidencia y límites','',
        '- [Reconstrucción: tabla completa](../../results/research/'+RECON+'/summary.csv).',
        '- [Pronóstico y persistencia filtrada](../../results/research/'+FORECAST+'/comparison_with_filtered_persistence.csv).',
        '- [Revisión científica independiente](SECOND_REVIEW.md).',
        '- [Viscoelasticidad 2D](VISCOELASTIC_2D.md) y [viscoplasticidad 2D](VISCOPLASTIC_2D.md).','',
        'Los solvers no se etiquetan como DNS universalmente certificada. La cuadratura de las leyes no polinómicas y los residuos constitutivos de proyección se conservan. '
        'El ruido temporal y la deriva de sensor siguen sin estimadores admisibles a los umbrales de reconstrucción: no se eliminan esos casos. '
        'No se ejecutaron estelas externas de cilindro EVP, simulaciones 3D, PIV real, GPU ni aprendizaje profundo. Las configuraciones correspondientes siguen como trabajo adicional.']
    (ROOT/'docs/research/EXTENDED_RESULTS.md').write_text('\n'.join(md)+'\n')
    latex=[r'\section{Hallazgos de la ampliación y controles}',
        f'En Taylor--Green gaussiano, TLS-DMD de rango cuatro pasa de {100*raw:.3f}\\% a {100*constrained:.3f}\\% de error mediano de velocidad con proyección y banda declarada. '
        f'En los tres casos newtonianos gaussianos, las condiciones aceptadas son {accepted_raw} sin proyección y {accepted_projection} con proyección. '
        'No se ajustó la banda utilizando la referencia limpia.',
        f'El dispatcher forward--backward a TLS utilizó su alternativa predeclarada en {mean_fallback} condiciones. Los fallos primarios permanecen en los registros y su trabajo cuenta en el costo total.',
        r'\section{Pronóstico cronológico ejecutado}',
        f'Se evaluaron {future_complete["conditions"]} condiciones con cinco campos, cuatro modelos de contaminación y tres semillas. '
        'Cada condición considera 29 opciones, con ajuste sólo en entrenamiento, selección por error de validación ruidosa y evaluación final en un bloque futuro independiente. '
        'Las opciones físicas comparten ajustes; no se cuentan como entrenamientos independientes.',
        f'La opción seleccionada mejora persistencia sin filtrar en {raw_gains} condiciones y persistencia filtrada en {gains}. '
        r'Se exige una diferencia absoluta mayor que $10^{-10}$ en el error relativo de velocidad; las diferencias de redondeo se consideran empates. '
        'Estos conteos son condicionales al banco y no probabilidades de éxito en mediciones reales.',
        r'\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{results/research/'+FORECAST+r'/figures/holdout_comparison.pdf}',
        r'\caption{Errores en el futuro excluido del ajuste. Comparar con persistencia filtrada evita atribuir toda mejora al modelo dinámico.}\end{figure}',
        r'\section{Banco reológico con condiciones iniciales comunes}',
        'Seis fluidos comparten velocidad inicial, densidad, dominio, malla y tiempos. Las diferencias numéricas representan cambios constitutivos; no son errores de reconstrucción. '
        'Las viscosidades HB se igualan en la tasa de corte declarada y las constantes viscoelásticas se comparan en su normalización nominal. '
        'La cercanía Oldroyd-B/FENE-P observada no demuestra capacidad de identificar materiales con velocidad ruidosa.',
        r'\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{results/research/rheology_matched_2026-10-14/final_vorticity.pdf}',
        r'\caption{Vorticidad final de seis modelos sobre el mismo estado inicial periódico. Las diferencias dependen de la ley constitutiva y su normalización declarada.}\end{figure}',
        r'\section{Cuestiones todavía abiertas}',
        'La vorticidad futura en casos de vórtices conserva errores importantes, y ninguna condición de ruido temporal o deriva de sensor satisface los umbrales de reconstrucción. '
        'La proyección reduce divergencia pero no recupera por sí sola sesgo de sensor, esfuerzos o memoria constitutiva. '
        'Los wakes externos, EVP, 3D, PIV, GPU y aprendizaje profundo siguen pendientes de ejecución.']
    (ROOT/'docs/research/extended_generated.tex').write_text('\n'.join(latex)+'\n')
    print(ROOT/'docs/research/EXTENDED_RESULTS.md')


if __name__=='__main__':main()
