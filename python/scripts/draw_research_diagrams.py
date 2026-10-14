#!/usr/bin/env python3
"""Diagrams for the research document; every field drawn here is schematic.

Run from any directory.  Output PDF files retain vector paths and selectable
text; PNG files are previews.  No simulation results are invented by this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INK = "#253548"
MUTED = "#546276"
BLUE = "#245c8a"
TEAL = "#167c80"
GOLD = "#ab761c"
RED = "#a6504b"
PALE_BLUE = "#edf4fa"
PALE_TEAL = "#edf7f5"
PALE_GOLD = "#fbf5e9"
PALE_RED = "#fbefee"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def canvas(title: str, subtitle: str, *, height: float = 6.4):
    fig, ax = plt.subplots(figsize=(11.8, height))
    fig.subplots_adjust(left=.018, right=.982, bottom=.025, top=.98)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.035, .962, title, fontsize=18, weight="bold", ha="left", va="top")
    ax.text(.035, .895, subtitle, fontsize=10.5, color=MUTED, ha="left", va="top")
    return fig, ax


def box(ax, x, y, w, h, title, body="", *, color=BLUE,
        fill=PALE_BLUE, title_size=12, body_size=10.5):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                          linewidth=1.1, edgecolor=color, facecolor=fill)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - .027, title, ha="center", va="top",
            fontsize=title_size, weight="bold", color=color)
    if body:
        ax.text(x + w / 2, y + h / 2 - .020, body, ha="center", va="center",
                fontsize=body_size, linespacing=1.55)
    return (x, y, w, h)


def arrow(ax, start, end, *, color=MUTED, curved=0., dashed=False,
          label=None, label_xy=None, width=1.5):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                color=color, linewidth=width,
                                connectionstyle=f"arc3,rad={curved}",
                                linestyle="--" if dashed else "-"))
    if label is not None:
        xy = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .025)
        ax.text(*xy, label, fontsize=10, ha="center", va="center", color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0})


def footer(ax, text):
    ax.text(.035, .025, text, ha="left", va="bottom", fontsize=9.5, color=MUTED,
            linespacing=1.4)


def constitutive_loop():
    fig, ax = canvas("La constitución cambia la dinámica del flujo",
                     "Dos acoplamientos reales del solver; variables adimensionales y dominio periódico.",
                     height=7.0)
    ax.text(.035, .825, "VISCOELASTICIDAD · memoria transportada y estirada", color=TEAL,
            fontsize=11.5, weight="bold")
    box(ax, .065, .568, .34, .198, "Cantidad de movimiento",
        "$\\partial_t\\mathbf{u}+\\mathbf{u}\\cdot\\nabla\\mathbf{u}$\n"
        "$=-\\nabla p+\\nu_s\\Delta\\mathbf{u}+\\nabla\\cdot\\boldsymbol{\\tau}_p+\\mathbf{F}$\n"
        "$\\nabla\\cdot\\mathbf{u}=0$", color=TEAL, fill=PALE_TEAL, body_size=11)
    box(ax, .585, .568, .34, .198, "Conformación polimérica $C$",
        "Advección + estiramiento\nRelajación hacia $C=I$\n"
        "$\\boldsymbol{\\tau}_p=(\\nu_p/\\lambda)\\,[f(C)C-I]$",
        color=TEAL, fill=PALE_TEAL, body_size=11)
    arrow(ax, (.412, .723), (.577, .723), color=TEAL, label="$\\nabla\\mathbf{u}$",
          label_xy=(.495, .764))
    arrow(ax, (.577, .602), (.412, .602), color=TEAL, label="$\\nabla\\cdot\\boldsymbol{\\tau}_p$",
          label_xy=(.495, .560))
    ax.text(.065, .49, "Oldroyd-B: $f=1$", fontsize=11)
    ax.text(.46, .49, "FENE-P: $f=(L^2-d)/(L^2-\\mathrm{tr}\\,C)$", fontsize=11)
    ax.text(.065, .442, "Velocidad planar; tensor constitutivo $3\\times3$ por defecto. "
            "Admisibilidad: $C\\succ0$ y, en FENE-P, $\\mathrm{tr}\\,C<L^2$.", fontsize=10.5)
    ax.text(.035, .365, "VISCOPLASTICIDAD · respuesta instantánea regularizada", color=GOLD,
            fontsize=11.5, weight="bold")
    box(ax, .065, .118, .34, .20, "Cantidad de movimiento",
        "Advección y presión\n"
        "$\\rho^{-1}\\nabla\\cdot(2\\eta_{\\rm app}\\mathbf{D})$\n"
        "$\\nabla\\cdot\\mathbf{u}=0$", color=GOLD, fill=PALE_GOLD, body_size=11)
    box(ax, .585, .118, .34, .20, "Herschel–Bulkley / Bingham",
        "$\\mathbf{D}=\\frac{1}{2}(\\nabla\\mathbf{u}+\\nabla\\mathbf{u}^{T})$\n"
        "$\\dot\\gamma=\\sqrt{2\\mathbf{D}:\\mathbf{D}}$\n"
        "$\\boldsymbol{\\tau}=2\\eta_{\\rm app}(\\dot\\gamma)\\mathbf{D}$",
        color=GOLD, fill=PALE_GOLD, body_size=11)
    arrow(ax, (.412, .271), (.577, .271), color=GOLD,
          label="deformación $\\mathbf{D}$", label_xy=(.495, .319))
    arrow(ax, (.577, .17), (.412, .17), color=GOLD,
          label="esfuerzo variable", label_xy=(.495, .117))
    footer(ax, "Bingham: $n=1$. La regularización suaviza la cedencia; no crea regiones rígidas exactas.\n"
           "La memoria viscoelástica y la viscosidad aparente no equivalen a reescalar una solución newtoniana.")
    return fig


def research_pipeline():
    fig, ax = canvas("Del flujo simulado a una comparación verificable",
                     "Reconstrucción: observaciones y geometría entran al estimador; el limpio se reserva para evaluar.",
                     height=7.2)
    x = [.045, .365, .685]
    w = .265
    box(ax, x[0], .655, w, .177, "1 · Simulación limpia",
        "Velocidad, tiempos y malla\nDensidad / esfuerzos disponibles\nParámetros + trazabilidad")
    box(ax, x[1], .655, w, .177, "2 · Contaminación",
        "Familia y semilla declaradas\nRMS sobre la ventana observada\n9 familias de ruido")
    box(ax, x[2], .655, w, .177, "3 · Estimación",
        "POD, RPCA y variantes DMD\nRangos / regularización declarados\nConvergencia y fallos visibles")
    arrow(ax, (.318, .742), (.357, .742))
    arrow(ax, (.638, .742), (.677, .742))
    box(ax, x[2], .312, w, .183, "4 · Restricción geométrica",
        "Helmholtz periódica, opcional\n+ banda estricta de 2/3, opcional\nSin consultar velocidad limpia",
        color=TEAL, fill=PALE_TEAL)
    arrow(ax, (.818, .647), (.818, .503))
    box(ax, x[1], .312, w, .183, "5 · Calidad y costo",
        "Error en $\\mathbf{u}$ y $\\omega$; balances\nCPU / tiempo / memoria / trabajo\nMediana e IQR de repeticiones",
        color=TEAL, fill=PALE_TEAL)
    box(ax, x[0], .312, w, .183, "6 · Comparación Pareto",
        "Mismo caso + ruido + semilla\nFiltrar admisibilidad y calidad\nConservar costos independientes",
        color=TEAL, fill=PALE_TEAL)
    arrow(ax, (.677, .404), (.638, .404), color=TEAL)
    arrow(ax, (.357, .404), (.318, .404), color=TEAL)
    # An independent reference branch never passes through the estimator.
    ax.plot([.176, .176, .497], [.647, .56, .56], color=GOLD, linestyle="--", lw=1.3)
    arrow(ax, (.497, .56), (.497, .503), color=GOLD, dashed=True)
    ax.text(.327, .582, "Referencia limpia → evaluación", color=GOLD, fontsize=10,
            ha="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    box(ax, .075, .115, .85, .108, "Dos usos de información que se registran por separado",
        "Costo instrumentado: ajuste + reconstrucción + proyección.   "
        "Evaluación física e I/O: fuera de ese intervalo.",
        color=MUTED, fill="#f5f6f8", body_size=10.5, title_size=11)
    footer(ax, "En pronóstico, la escala de ruido usa sólo entrenamiento; "
           "la selección usa validación ruidosa (véase el diagrama cronológico).")
    return fig


def chronological_protocol():
    fig, ax = canvas("Pronosticar exige separar pasado, selección y futuro",
                     "Bloques cronológicos sin barajado ni solapamiento; ningún dato de prueba ajusta el modelo.",
                     height=6.7)
    left, total = .065, .87
    parts = [(0, .6, "Entrenamiento · 60 %", BLUE, PALE_BLUE),
             (.6, .2, "Validación\n20 %", GOLD, PALE_GOLD),
             (.8, .2, "Prueba\n20 %", TEAL, PALE_TEAL)]
    for a, width, label, color, fill in parts:
        xx = left + total * a
        ax.add_patch(Rectangle((xx, .717), total * width, .101, edgecolor=color,
                               facecolor=fill, linewidth=1.2))
        ax.text(xx + total * width / 2, .766, label, ha="center", va="center",
                color=color, weight="bold", fontsize=11)
    arrow(ax, (.065, .864), (.936, .864), label="tiempo creciente",
          label_xy=(.50, .864), width=1.)
    box(ax, .065, .419, .36, .20, "Ajustar con observaciones",
        "RPCA / Stable RPCA: sólo este bloque\n"
        "Bases, media y dinámica: sólo pasado\n"
        "Ruido: escala calibrada aquí", body_size=10.5)
    box(ax, .462, .419, .22, .20, "Elegir el candidato",
        "Predicción vs. observación\nruidosa de validación\n"
        "Minimizar MSE observado", color=GOLD, fill=PALE_GOLD, body_size=10.2)
    box(ax, .721, .419, .215, .20, "Medir el resultado",
        "Predicción vs. referencia\nlimpia de prueba\n"
        "Errores y métricas físicas", color=TEAL, fill=PALE_TEAL, body_size=10.2)
    arrow(ax, (.24, .709), (.24, .627), color=BLUE)
    arrow(ax, (.665, .709), (.573, .627), color=GOLD)
    arrow(ax, (.849, .709), (.829, .627), color=TEAL)
    arrow(ax, (.433, .53), (.454, .53))
    arrow(ax, (.692, .53), (.713, .53))
    ax.text(.49, .351, "Congelar la elección antes de calcular el error limpio de prueba", fontsize=11,
            ha="center", weight="bold", color=TEAL)
    box(ax, .065, .132, .87, .147, "Comparadores sometidos al mismo protocolo",
        "Persistencia del último estado de entrenamiento; persistencia + la misma proyección física.\n"
        "Sin refit tras seleccionar; sin recorte de crecimiento estimado; un fallo permanece como fallo.",
        color=MUTED, fill="#f5f6f8", title_size=11, body_size=10.5)
    footer(ax, "Fracciones enteras: $N_{\\rm train}=\\lfloor0.6T\\rfloor$, "
           "$N_{\\rm val}=\\lfloor0.2T\\rfloor$; prueba conserva los tiempos restantes.\n"
           "Las observaciones ruidosas de prueba no se leen al ajustar, seleccionar ni predecir.")
    return fig


def spectral_resolution():
    fig, ax = canvas("Mismo dominio, distinta capacidad de resolver escalas",
                     "ESQUEMA de mallas 64² / 128² / 256²: no son campos simulados ni una prueba de convergencia.",
                     height=7.2)
    centers = [.185, .5, .815]
    yy, side = .463, .207
    grid_height = side * 11.8 / 7.2
    for n, cx in zip([64, 128, 256], centers):
        xx = cx - side / 2
        ax.text(cx, .816, f"{n} × {n} nodos", fontsize=13, color=BLUE,
                weight="bold", ha="center")
        # Every eighth mesh line is displayed, for legibility in the document.
        count = n // 8
        for a in np.linspace(0, 1, count + 1):
            ax.plot([xx + a * side, xx + a * side], [yy, yy + grid_height], color="#bacbda", lw=.55)
            ax.plot([xx, xx + side], [yy + a * grid_height, yy + a * grid_height], color="#bacbda", lw=.55)
        ax.add_patch(Rectangle((xx, yy), side, grid_height, fill=False, edgecolor=BLUE, lw=1.2))
        # This is a schematic fixed physical scale, not an estimated vortex core.
        ax.add_patch(Ellipse((cx, yy + grid_height / 2), side * .36, grid_height * .36,
                            fill=False, edgecolor=GOLD, linewidth=1.6, zorder=3))
        arrow(ax, (cx - side * .18, yy + grid_height / 2),
              (cx + side * .18, yy + grid_height / 2), color=GOLD, width=1.)
        ax.text(cx, yy + grid_height / 2 + .079, "$w$", ha="center", color=GOLD, fontsize=12)
        ax.text(cx, .414, f"$\\Delta x=L/{n}$", ha="center", fontsize=12)
        ax.text(cx, .367, f"$\\Delta x/w=L/({n}w)$", ha="center", fontsize=11)
        cut = (n - 1) // 3
        ax.plot([cx - .12, cx + .12], [.289, .289], color=MUTED, lw=1.)
        ratio = cut / (n / 2)
        ax.plot([cx - .12 * ratio, cx + .12 * ratio], [.289, .289], color=TEAL, lw=6,
                solid_capstyle="butt")
        ax.plot([cx - .12, cx - .12], [.276, .302], color=RED, lw=1.1)
        ax.plot([cx + .12, cx + .12], [.276, .302], color=RED, lw=1.1)
        ax.text(cx, .335, f"Modos retenidos: $|k|\\leq {cut}$", ha="center", fontsize=10.5,
                color=TEAL)
        ax.text(cx, .247, f"Nyquist: $|k|={n // 2}$", ha="center", fontsize=10, color=RED)
        ax.text(cx, .197, f"$\\lambda_{{\\min,\\rm eje}}=L/{cut}$", ha="center", fontsize=10.5)
    ax.text(.50, .117, "Corte estricto por eje: $|k_x|<N_x/3$, $|k_y|<N_y/3$ "
            "(índices de Fourier enteros)", ha="center", fontsize=11, color=TEAL)
    footer(ax, "Se dibuja una línea de malla de cada ocho. El círculo representa una escala física $w$ fija, sin valor estimado.\n"
           "Aumentar $N$ sólo aporta información si la dinámica ocupa esas escalas y los diagnósticos convergen.")
    return fig


def independent_checks():
    fig, ax = canvas("Una malla grande necesita controles independientes",
                     "La evidencia se construye cambiando una fuente de error cada vez, con geometría y régimen declarados.",
                     height=7.5)
    rows = [
        (.698, BLUE, PALE_BLUE, "Resolución espacial", "$N$, $2N$, $4N$",
         "Mantener modelo y paso suficientemente pequeño",
         "Campos, gradientes y cola espectral"),
        (.542, TEAL, PALE_TEAL, "Integración temporal", "$\\Delta t$, $\\Delta t/2$, $\\Delta t/4$",
         "Mantener malla; comprobar pasos realmente aceptados",
         "Orden observado y error de muestreo"),
        (.386, GOLD, PALE_GOLD, "Definición constitutiva", "$m$, $\\epsilon$; $\\kappa$ si se usa",
         "Separar regularización de cedencia y difusión añadida",
         "Sensibilidad del modelo físico"),
        (.230, RED, PALE_RED, "Cuadratura / proyección", "padding 1.5×, 2×, 3×",
         "Malla del estado fija; evaluar términos no polinomiales",
         "Aliasing residual y defecto energético"),
    ]
    for y, color, fill, title, control, held, evidence in rows:
        box(ax, .047, y, .245, .12, title, control, color=color, fill=fill,
            title_size=11.5, body_size=11)
        arrow(ax, (.3, y + .06), (.345, y + .06), color=color)
        ax.text(.36, y + .077, held, fontsize=10.5, va="center")
        ax.text(.36, y + .025, "Evaluar: " + evidence, fontsize=10.5, color=color, va="center")
    ax.text(.50, .135, "En todos los controles: finitud, admisibilidad, balance de energía y divergencia", ha="center",
            fontsize=11, color=INK, weight="bold")
    footer(ax, "Comparar con referencias analíticas cuando existen; con referencias numéricas refinadas en el resto.\n"
           "La convergencia respalda ese régimen y ese tiempo; no certifica DNS, caos, estelas ni todos los parámetros.")
    return fig


def matrix_methods():
    fig, ax = canvas("Dos componentes espaciales, una secuencia temporal",
                     "ESQUEMA de organización de datos; los colores de matrices ilustran estructura, no resultados medidos.",
                     height=7.4)
    ax.text(.035, .819, "Instantáneas $\\mathbf{u}(t_j)=(u_x,u_y)$", fontsize=12, weight="bold", color=BLUE)
    # Smooth illustrative array, deliberately not presented as a simulated flow.
    xx, yy = np.meshgrid(np.linspace(0, 2*np.pi, 12), np.linspace(0, 2*np.pi, 12))
    field = np.sin(xx) * np.cos(yy)
    for a, x in enumerate([.055, .166, .277]):
        ax.imshow(np.roll(field, a, axis=0), extent=(x, x + .087, .648, .774),
                  origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
        ax.text(x + .0435, .617, "$t_" + str(a) + "$", ha="center", fontsize=11)
    ax.text(.06, .567, "Vectorizar $u_x$ y $u_y$\napilados en cada columna", fontsize=10.5)
    arrow(ax, (.386, .70), (.444, .70), label="apilar", label_xy=(.415, .757))
    # Exactly two equal feature blocks, not an incorrect space/time orientation.
    arr = np.linspace(-1, 1, 120).reshape(12, 10)
    ax.imshow(arr, extent=(.474, .688, .594, .787), cmap="RdBu_r", origin="upper",
              interpolation="nearest", aspect="auto", vmin=-1, vmax=1)
    ax.plot([.474, .688], [.6905, .6905], color="white", lw=2)
    ax.text(.455, .744, "$u_x$", ha="right", fontsize=11)
    ax.text(.455, .644, "$u_y$", ha="right", fontsize=11)
    ax.text(.58, .819, "$X=[\\mathbf{x}_0,\\ldots,\\mathbf{x}_{T-1}]$", ha="center", fontsize=12)
    ax.text(.58, .554, "$X\\in\\mathbb{R}^{(2N_xN_y)\\times T}$", ha="center", fontsize=11)
    arrow(ax, (.70, .69), (.75, .69))
    box(ax, .774, .61, .176, .174, "Pares temporales",
        "$X_-=[\\mathbf{x}_0,\\ldots,\\mathbf{x}_{T-2}]$\n"
        "$X_+=[\\mathbf{x}_1,\\ldots,\\mathbf{x}_{T-1}]$",
        body_size=10.5, title_size=11)
    items = [
        (.048, BLUE, PALE_BLUE, "POD", "$X\\approx U_r\\Sigma_rV_r^T$",
         "Base espacial de baja dimensión\nCompresión / reconstrucción\nNo define por sí sola evolución futura"),
        (.367, TEAL, PALE_TEAL, "DMD", "$X_+\\approx A X_-$",
         "Modos + tasas temporales\nAjuste reducido de dinámica\nEl ruido altera operador y autovalores"),
        (.686, GOLD, PALE_GOLD, "RPCA", "$X=L+S$ (PCP)",
         "Bajo rango + errores dispersos\nStable RPCA admite residuo denso\nNo identifica todo ruido coherente"),
    ]
    for x, color, fill, title, eq, body in items:
        box(ax, x, .201, .264, .258, title, eq + "\n\n" + body,
            color=color, fill=fill, title_size=12, body_size=10.2)
    footer(ax, "La implementación excluye sólidos usando una máscara conocida, si existen; las formas anteriores ilustran el caso periódico sin sólidos.\n"
           "RPCA → DMD: preprocesar antes del modelo dinámico; en pronóstico, ambas operaciones usan sólo entrenamiento.")
    return fig


DIAGRAMS = {
    "diagram_constitutive_loop": constitutive_loop,
    "diagram_research_pipeline": research_pipeline,
    "diagram_chronological_protocol": chronological_protocol,
    "diagram_spectral_resolution": spectral_resolution,
    "diagram_independent_checks": independent_checks,
    "diagram_matrix_methods": matrix_methods,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/research/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, build in DIAGRAMS.items():
        fig = build()
        fig.savefig(args.output_dir / f"{name}.pdf", dpi=220)
        fig.savefig(args.output_dir / f"{name}.png", dpi=180)
        plt.close(fig)
        print(name)


if __name__ == "__main__":
    main()
