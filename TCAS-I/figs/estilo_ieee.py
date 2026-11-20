"""Estilo compartido para las figuras del manuscrito TCAS-I.

Tipografia LaTeX real: backend pgf + text.usetex, con EL MISMO preambulo de
fuentes que IEEEtran para que las figuras sean indistinguibles del cuerpo.

Comprobado con `pdffonts main.pdf`: IEEEtran compone el texto con
NimbusRomNo9L (Times = familia `ptm`, cargada por ot1ptm.fd) y DEJA LAS
MATEMATICAS EN COMPUTER MODERN (CMMI/CMSY/CMR/CMEX). Por eso aqui se carga
`times` a secas y NO newtxmath: con newtx las figuras salian en
TeXGyreTermes + NewTXMI, que se parece a Times pero no es la misma fuente y
canta al lado del cuerpo. Verificar tras cada cambio con:

    pdffonts figs/*.pdf      # debe dar NimbusRomNo9L + CM*, nada de newtx
"""
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib.text import Text as _Text

PREAMBLE = "\n".join([
    r"\usepackage{times}",   # texto Times/ptm, matematicas en CM: igual que IEEEtran
    r"\usepackage{amsmath,amssymb,bm}",
])

plt.rcParams.update({
    # --- LaTeX ---
    'text.usetex': True,
    'pgf.texsystem': 'pdflatex',
    'pgf.rcfonts': False,
    'pgf.preamble': PREAMBLE,
    'text.latex.preamble': PREAMBLE,
    'font.family': 'serif',
    # --- tamanos IEEE (cuerpo 10pt, figuras 8pt) ---
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    # --- trazos finos para columna estrecha ---
    'axes.linewidth': 0.6, 'grid.linewidth': 0.4, 'lines.linewidth': 0.9,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'legend.framealpha': 0.92, 'legend.edgecolor': '0.75',
    'legend.borderpad': 0.3, 'legend.labelspacing': 0.3,
    'legend.handlelength': 1.6, 'legend.handletextpad': 0.4,
    'savefig.dpi': 600, 'figure.dpi': 600,
})

# Ancho de columna IEEE en pulgadas
COLW = 3.487


def comprueba_solapes(fig, tol=0.5, verbose=True):
    """Detecta solapamientos entre TODOS los textos de la figura.

    Recorre titulos, rotulos de eje, ticklabels, anotaciones y leyendas,
    calcula sus bounding boxes en pixeles sobre el canvas ya renderizado y
    reporta cada par que se cruza mas de `tol` px. Devuelve la lista de
    colisiones, para poder usarla como asercion en el script de la figura.

    Solo se comparan textos que compartan region: los de ejes distintos
    tambien se comparan, porque en una figura multipanel el solape tipico es
    justo entre paneles (leyenda de figura contra xlabel de un panel).
    """
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    items = []

    def anota(art, etiq):
        try:
            txt = art.get_text()
        except Exception:
            txt = etiq
        if not str(txt).strip():
            return
        try:
            # Annotation.get_window_extent incluye LA FLECHA, no solo los
            # glifos: dos rotulos separados media figura dan solape si sus
            # flechas se cruzan. Aqui interesa el texto, asi que se llama al
            # metodo de Text saltandose la sobrecarga de Annotation.
            bb = _Text.get_window_extent(art, renderer=ren)
        except Exception:
            try:
                bb = art.get_window_extent(renderer=ren)
            except Exception:
                return
        if bb.width <= 0 or bb.height <= 0:
            return
        items.append((etiq, str(txt)[:38], bb))

    def ticks_visibles(ax, eje):
        """Ticklabels realmente dibujados: matplotlib conserva los del
        localizador que caen fuera de los limites, y contarlos genera falsos
        positivos (dos ejes 'chocando' por etiquetas invisibles)."""
        # Un eje con axis('off') conserva sus ticklabels como objetos aunque
        # no se dibujen: contarlos da falsos positivos en paneles esquematicos.
        if not getattr(ax, 'axison', True):
            return []
        # Un eje entero puede estar oculto sin estarlo el resto: es el caso de
        # twinx/twiny, que crea un Axes nuevo cuyo eje x se apaga con
        # xaxis.set_visible(False) para no redibujar los ticks del original.
        # Sus ticklabels siguen existiendo como objetos con bbox.
        if not (ax.xaxis if eje == 'x' else ax.yaxis).get_visible():
            return []
        if eje == 'x':
            locs, labels, (lo, hi) = ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()
        else:
            locs, labels, (lo, hi) = ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim()
        lo, hi = min(lo, hi), max(lo, hi)
        span = hi - lo
        out = []
        for loc, lab in zip(locs, labels):
            if not lab.get_visible():
                continue
            if lo - 1e-9*span <= loc <= hi + 1e-9*span:
                out.append(lab)
        return out

    for i, ax in enumerate(fig.axes):
        anota(ax.title, f'ax{i}.title')
        anota(ax.xaxis.label, f'ax{i}.xlabel')
        anota(ax.yaxis.label, f'ax{i}.ylabel')
        for j, tl in enumerate(ticks_visibles(ax, 'x')):
            anota(tl, f'ax{i}.xtick{j}')
        for j, tl in enumerate(ticks_visibles(ax, 'y')):
            anota(tl, f'ax{i}.ytick{j}')
        for j, tx in enumerate(ax.texts):
            anota(tx, f'ax{i}.text{j}')
        leg = ax.get_legend()
        if leg is not None:
            try:
                items.append((f'ax{i}.legend', '<legend>',
                              leg.get_window_extent(renderer=ren)))
            except Exception:
                pass
    for j, tx in enumerate(fig.texts):
        anota(tx, f'fig.text{j}')
    for j, leg in enumerate(getattr(fig, 'legends', [])):
        try:
            items.append((f'fig.legend{j}', '<figlegend>',
                          leg.get_window_extent(renderer=ren)))
        except Exception:
            pass

    choques = []
    for a in range(len(items)):
        for b in range(a+1, len(items)):
            (na, ta, ba), (nb, tb, bb) = items[a], items[b]
            inter = ba.intersection(ba, bb)
            if inter is None:
                continue
            if inter.width > tol and inter.height > tol:
                choques.append((na, ta, nb, tb, inter.width, inter.height))

    if verbose:
        if choques:
            print(f'  !! {len(choques)} SOLAPES DE TEXTO:')
            for na, ta, nb, tb, w, h in choques:
                print(f'     {na} ("{ta}")  x  {nb} ("{tb}")  '
                      f'-> {w:.1f} x {h:.1f} px')
        else:
            print(f'  sin solapes de texto ({len(items)} elementos revisados)')
    return choques
