import bisect
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib as mpl
from io import BytesIO
import base64
mpl.use('agg')
pd.options.mode.chained_assignment = None  # default='warn'

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta



url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRVqjkY-61m5LWel2zI_1bGPG4a5CHOvUwCobn1LqWuACg4" \
      "-sYtvaQDxqUre33J5graTPsAY_sksLNC/pubhtml "


def get_disc_graf(id):
    disc = list_disc_for_graf(id)
    label = ['Dominante', 'Influyente', 'Concienzudo', 'Estable']
    graf = grafico_bar_alt(label, disc, "Perfil")
    return graf


def get_disc_word(id):
    disc = list_disc_for_graf(id)
    label = ['Dominante', 'Influyente', 'Concienzudo', 'Estable']
    graf = grafico_bar(label, disc, "Perfil")
    return graf


def grafico_bar_alt(labels, values, title=""):
    x = labels
    y = values
    colores = ["#619cff", "#00ba38", "#f8766d", "darkseagreen"]
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.subplots_adjust(top=0.90, bottom=0.25, left=0.11, right=0.85)
    ax.set_title(title,  position=(0.5, 1.1), ha='center')
    ax.barh(x, width = y, color=colores)
    list_patch = []
    for i in range(len(labels)):
        patch = mpatches.Patch(color=colores[i], label=labels[i])
        list_patch.append(patch)
    ax.legend(handles=list_patch, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,str(round((i.get_width()))),
                 fontsize=10, fontweight='bold',color='grey')
    ax.invert_yaxis()
    ax.set_xlim([0,150])
    
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.xticks([])
    plt.yticks([])
    buffer = BytesIO()
    plt.savefig(buffer, format='png',transparent=True)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    return graphic


def grafico_bar(labels, values, title=""):
    x = labels
    y = values
    colores = ["#619cff", "#00ba38", "#f8766d", "darkseagreen"]
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(top=0.90, bottom=0.45, left=0.11, right=0.85)
    ax.set_title(title,  position=(0.5, 1.1), ha='center')
    ax.barh(x, width = y, color=colores)
    list_patch = []
    for i in range(len(labels)):
        patch = mpatches.Patch(color=colores[i], label=labels[i])
        list_patch.append(patch)
    ax.legend(handles=list_patch, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,str(round((i.get_width()))),
                 fontsize=10, fontweight='bold',color='grey')
    ax.invert_yaxis()
    ax.set_xlim([0,150])
    
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.xticks([])
    plt.yticks([])
    buffer = BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return image_png

def get_grafico_polar_liderazgo_word(id):
    lider = list_lider_for_graf(id)
    label = ['Entusiasmo', 'Integridad', 'Autorenovacion', 'Fortaleza', 'Percepción', 'Criterio', 'Ejecución', 'Audacia', 'Construcción \n de un equipo', 'Colaboración', 'Inspiración', 'Servir a\n los demás']
    graf = grafico_polar_alt(label, lider, "Dimensión del liderazgo")
    return graf


def get_grafico_polar_care_word(id):
    care = list_care_for_graf(id)
    label = ['Conceptual', 'Espontáneo', 'Normativo', 'Metódico']
    graf = grafico_polar_alt(label, care)
    return graf


def get_grafico_polar_liderazgo_render(id):
    lider = list_lider_for_graf(id)
    label = ['Entusiasmo', 'Integridad', 'Autorenovacion', 'Fortaleza', 'Percepción', 'Criterio', 'Ejecución', 'Audacia', 'Construcción \n de un equipo', 'Colaboración', 'Inspiración', 'Servir a \n los demás']
    graf = grafico_polar(label, lider, "Dimensión del liderazgo")
    return graf


def get_grafico_polar_care_render(id):
    care = list_care_for_graf(id)
    label = ['Conceptual', 'Espontáneo', 'Normativo', 'Metódico']
    graf = grafico_polar(label, care)
    return graf


def grafico_polar_alt(label, values, titulo=" "):
    data = [label,
        (titulo, [
            values])]

    N = len(data[0])
    theta = radar_factory(N, frame='polygon')

    spoke_labels = data.pop(0)
    title, case_data = data[0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.82)

    for d in case_data:
        line = ax.plot(theta, d)
        ax.fill(theta, d,  alpha=0.25)
        ax.scatter(theta, d, color='crimson', s=10)
    ax.set_varlabels(spoke_labels)
    ax.tick_params(axis='x',pad=20)
    if len(label) > 4:
        ax.set_rgrids([0, 10, 20])
        ax.set_ylim([0,25])
        ax.tick_params(axis='both',pad=31, labelsize=13, direction='out')
        plt.gcf().text(0.3, 0.93, "Dimensión del liderazgo", rotation = 0, fontsize=14)
        plt.gcf().text(0.65, 0.12, "Realización", rotation = 30, fontsize=14)
        plt.gcf().text(0.09, 0.14, "Análisis", rotation = -30, fontsize=14)
        plt.gcf().text(0.08, 0.85, "Caracter", rotation = 30, fontsize=14)
        plt.gcf().text(0.82, 0.76, "Interacción", rotation = -40, fontsize=14)
        fig.subplots_adjust(top=0.95, bottom=0.1, left=0.18, right=0.72)
    else:
        ax.set_rgrids([0])
        ax.set_ylim([0,71])
        ax.set_title(title,  position=(0.5, 1.1), ha='center')
        ax.tick_params(axis='x',pad=28, labelsize=13)
        fig.subplots_adjust(top=0.95, bottom=0.1, left=0.19, right=0.82)
    buffer = BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return image_png


def grafico_polar(label, values, titulo=" "):
    data = [label,
        (titulo, [
            values])]

    N = len(data[0])
    theta = radar_factory(N, frame='polygon')

    spoke_labels = data.pop(0)
    title, case_data = data[0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.82)

    
    ax.set_title(title,  position=(0.5, 1.1), ha='center')

    for d in case_data:
        line = ax.plot(theta, d)
        ax.fill(theta, d,  alpha=0.25)
        ax.scatter(theta, d, color='crimson', s=10)
    ax.set_varlabels(spoke_labels)
    
    
    
    if len(label) > 4:
        ax.set_rgrids([0, 10, 20])
        ax.set_ylim([0,25])
        ax.tick_params(axis='x',pad=24)
        plt.gcf().text(0.65, 0.05, "Realización", rotation = 30, fontsize=14)
        plt.gcf().text(0.06, 0.14, "Análisis", rotation = -30, fontsize=14)
        plt.gcf().text(0.08, 0.85, "Caracter", rotation = 30, fontsize=14)
        plt.gcf().text(0.82, 0.76, "Interacción", rotation = -40, fontsize=14)
    else:
        ax.set_rgrids([0])
        ax.set_ylim([0,71])
        ax.tick_params(axis='x',pad=20)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    
    return graphic
    """ N = 4
    theta = radar_factory(N, frame='polygon')
    data = list_care_for_graf(id)
    fig, axs = plt.subplots(figsize=(9, 9), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    colors = ['b', 'r', 'g', 'm']
    spoke_labels = ['Conceptual', 'Espontáneo', 'Normativo', 'Metódico']
    # Plot the four cases from the example data on separate Axes
    for ax, (case_data) in zip(axs.flat, data):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title("Perfil", weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels)
    fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.show() """


def cargar_dataframe(url):

    # Se lee la página web, el argumento header=1 indica que el nombre de las columnas está en la segunda fila
    # El encoding="UTF-8" asegura que se reconozca los acentos y la ñ
    tablas = pd.read_html(url, header=1, encoding="UTF-8")
    df = tablas[0]
    return df

df_disc = cargar_dataframe(url)


def set_dicen(x):
    list_d = []
    opciones_mas_dicen = ['z', 'c', 'e', 't', 'n']
    opciones_menos_dicen = ['z-', 'c-', 'e-', 't-', 'n-']
    c = ['alegre', 'alentador/a', 'amable', 'amigable', 'anima a los demás', 'animado/a', 'cautivador/a', 'comunicativo/a', 'convincente', 'de trato fácil', 'desenvuelto/a', 'encantador/a', 'entusiasta', 'espontáneo/a', 'estimulante', 'expresivo/a', 'expresivo/a', 'extrovertido/a', 'impetuoso/a', 'impulsivo/a', 'ingenioso/a', 'jovial', 'popular', 'promotor/a', 'receptivo/a', 'sociable', 'sociable', 'vivaz']
    z = ['acepta riesgos', 'agresivo/a', 'atrevido/a', 'audaz', 'autosuficiente', 'competitivo/a', 'decidido/a', 'decisivo/a', 'directo/a', 'dominante', 'enérgico/a', 'exigente', 'franco/a', 'habla directo', 'ideas firmes', 'impaciente', 'independiente', 'inquieto/a', 'insistente', 'osado/a', 'persistente', 'pionero/a', 'rápido/a', 'resuelto/a', 'tenaz', 'vigoroso/a', 'le agrada discutir']
    e = ['analítico/a', 'apegado a las normas', 'cauteloso/a', 'cauto/a', 'certero/a', 'controlado/a', 'cuida los detalles', 'cuidadoso/a', 'discernimiento', 'discreto/a', 'evaluador/a', 'investigador/a', 'lógico/a', 'meticuloso/a', 'metódico/a', 'perceptivo/a', 'perfeccionista', 'precavido/a', 'preciso/a', 'prevenido/a', 'prudente', 'reflexivo/a', 'reservado/a', 'sistemático/a', 'viváz']
    t = ['adaptable', 'amable', 'ameno/a', 'amistoso/a', 'apacible', 'atento/a', 'bondadoso/a', 'calmado/a', 'colaborador/a', 'compasivo/a', 'complaciente', 'considerado/a', 'constante', 'contento/a', 'cortés', 'equilibrado', 'generoso/a', 'gentil', 'leal', 'moderado/a', 'obediente', 'paciente', 'pacifico/a', 'sensible', 'tolerante', 'tolerante', 'tranquilo/a', 'tranquilo/a']
    n = ['valeroso/a']
    if x in c:
        return 1
    elif x in z:
        return 2
    elif x in e:
        return 3
    elif x in t:
        return 4
    elif x in n:
        return 5
    else:
        return 0


def set_dicen2(x):
    c = ['alegre', 'alentador/a', 'amable', 'amigable', 'anima a los demás', 'animado/a', 'cautivador/a', 'comunicativo/a', 'convincente', 'de trato fácil', 'desenvuelto/a', 'encantador/a', 'entusiasta', 'espontáneo/a', 'estimulante', 'expresivo/a', 'extrovertido/a', 'impetuoso/a', 'impulsivo/a', 'ingenioso/a', 'jovial', 'popular', 'promotor/a', 'receptivo/a', 'sociable', 'vivaz']
    z = ['valeroso/a','acepta riesgos', 'agresivo/a', 'atrevido/a', 'audaz', 'autosuficiente', 'competitivo/a', 'decidido/a', 'decisivo/a', 'directo/a', 'dominante', 'enérgico/a', 'exigente', 'franco/a', 'habla directo', 'ideas firmes', 'impaciente', 'independiente', 'inquieto/a', 'insistente', 'osado/a', 'persistente', 'pionero/a', 'rápido/a', 'resuelto/a', 'tenaz', 'vigoroso/a', 'le agrada discutir']
    e = ['analítico/a', 'apegado a las normas', 'cauteloso/a', 'cauto/a', 'certero/a', 'controlado/a', 'cuida los detalles', 'cuidadoso/a', 'discernimiento', 'discreto/a', 'evaluador/a', 'investigador/a', 'lógico/a', 'meticuloso/a', 'metódico/a', 'perceptivo/a', 'perfeccionista', 'precavido/a', 'preciso/a', 'prevenido/a', 'prudente', 'reflexivo/a', 'reservado/a', 'sistemático/a', 'viváz']
    t = ['adaptable', 'amable', 'ameno/a', 'amistoso/a', 'apacible', 'atento/a', 'bondadoso/a', 'calmado/a', 'colaborador/a', 'compasivo/a', 'complaciente', 'considerado/a', 'constante', 'contento/a', 'cortés', 'equilibrado', 'generoso/a', 'gentil', 'moderado/a', 'obediente', 'paciente', 'pacifico/a', 'sensible', 'tolerante', 'tolerante', 'tranquilo/a', 'tranquilo/a']
    m = ['leal']
    if x in c:
        return 1
    elif x in z:
        return 2
    elif x in e:
        return 3
    elif x in t:
        return 4
    elif x in m:
        return 5
    else:
        return 0
    

def set_dicen3(x):
    c = ['alegre', 'alentador/a', 'amable', 'amigable', 'anima a los demás', 'animado/a', 'cautivador/a', 'comunicativo/a', 'convincente', 'de trato fácil', 'desenvuelto/a', 'encantador/a', 'entusiasta', 'espontáneo/a', 'estimulante', 'expresivo/a', 'extrovertido/a', 'impetuoso/a', 'impulsivo/a', 'ingenioso/a', 'jovial', 'popular', 'promotor/a', 'receptivo/a', 'sociable']
    z = ['valeroso/a','acepta riesgos', 'agresivo/a', 'atrevido/a', 'audaz', 'autosuficiente', 'competitivo/a', 'decidido/a', 'decisivo/a', 'directo/a', 'dominante', 'enérgico/a', 'exigente', 'franco/a', 'habla directo', 'ideas firmes', 'impaciente', 'independiente', 'inquieto/a', 'insistente', 'osado/a', 'persistente', 'pionero/a', 'rápido/a', 'resuelto/a', 'tenaz', 'vigoroso/a', 'le agrada discutir']
    e = ['analítico/a', 'apegado a las normas', 'cauteloso/a', 'cauto/a', 'certero/a', 'controlado/a', 'cuida los detalles', 'cuidadoso/a', 'discernimiento', 'discreto/a', 'evaluador/a', 'investigador/a', 'lógico/a', 'meticuloso/a', 'metódico/a', 'perceptivo/a', 'perfeccionista', 'precavido/a', 'preciso/a', 'prevenido/a', 'prudente', 'reflexivo/a', 'reservado/a', 'sistemático/a', 'vivaz']
    t = ['adaptable', 'amable', 'ameno/a', 'amistoso/a', 'apacible', 'atento/a', 'bondadoso/a', 'calmado/a', 'colaborador/a', 'compasivo/a', 'complaciente', 'considerado/a', 'constante', 'contento/a', 'cortés', 'equilibrado', 'generoso/a', 'gentil', 'moderado/a', 'obediente', 'paciente', 'pacifico/a', 'sensible', 'tolerante', 'tolerante', 'tranquilo/a', 'tranquilo/a']
    m = ['leal']
    if x in c:
        return 1
    elif x in z:
        return 2
    elif x in e:
        return 3
    elif x in t:
        return 4
    elif x in m:
        return 5
    else:
        return 0

var = """ clinico = ["Agresividad", "Ansiedad", "Atipicidad", "Depresion", "Hiperactividad", "Problemas de atencion",
                   "Retraimiento", "Somatizacion", "Problemas de conducta", "Problemas de aprendizaje",
                   "Actitud negativa hacia el colegio", "Actitud negativa hacia los profesores", "Locus de control",
                   "Estres social", "Sentido de incapacidad", "Busqueda de sensaciones"]
        adaptable = ["Adaptabilidad", "Habilidades sociales"]
        condiciones = [df_final[f'T {dimension}'] <= 30, df_final[f'T {dimension}'] <= 40, df_final[f'T {dimension}'] <= 59,
                       df_final[f'T {dimension}'] <= 69, df_final[f'T {dimension}'] <= 129]
        if dimension in clinico:
            df_final[f'Nivel {dimension}'] = np.select(condiciones, opciones_niveles_clinico)
        else:
            df_final[f'Nivel {dimension}'] = np.select(condiciones, opciones_niveles_adapta)"""


def percentil_dicen(valores, baremo, columna_comparar, columna_recuperar):
    resultado = []
    for j in range(len(valores)):
        if baremo == 'A':
            df2 = pd.read_pickle('../baremos/disc_a.pkl')
        elif baremo == 'B':
            df2 = pd.read_pickle('../baremos/disc_b.pkl')
        else:
            df2 = pd.read_pickle('../baremos/disc_c.pkl')

        df2 = df2.loc[:, [columna_recuperar, columna_comparar]]
        df2 = df2.dropna()
        
        sintomas = df2.loc[:, columna_comparar].values.tolist()
        sintomas = sorted(sintomas)
        pc = df2.loc[:, columna_recuperar].values.tolist()
        if baremo == 'B':
            pc = sorted(pc, reverse=True)
        else:
            pc = sorted(pc)
        i = bisect.bisect_left(sintomas, valores[j])
        resultado.append(int(pc[i]))
    return resultado


def result_dicen(valores, baremo, columna_comparar, columna_recuperar):
    resultado = []
    for j in range(len(valores)):
        df2 = pd.read_pickle(f'../baremos/{baremo}.pkl')
        try:
            df2 = df2.loc[df2[columna_comparar] == int(valores[j]), [columna_recuperar]]
        except ValueError:
            df2 = df2.loc[df2[columna_comparar] == valores[j], [columna_recuperar]]
        df2 = df2.values.tolist()
        try:
            resultado.append(df2[0][0])
        except IndexError:
            resultado.append("-")
    
    return resultado


def result_dicen_alternativo(valor, baremo, columna_comparar, columna_recuperar):
    df2 = pd.read_pickle(f'../baremos/{baremo}.pkl')
    try:
        df2 = df2.loc[df2[columna_comparar] == int(valor), [columna_recuperar]]
    except ValueError:
        df2 = df2.loc[df2[columna_comparar] == valor, [columna_recuperar]]
    df2 = df2.values.tolist()
    try:
        df2 = df2[0][0]
    except IndexError:
        df2 = "-"
    return df2


def info_test_total(id):
    #df = cargar_dataframe(url)
    df_info = df_disc.iloc[[id]]
    df_temp = df_info.iloc[:, -3]
    df_info = df_info.iloc[:, :10]
    df_info['Edad'] = df_info['Edad'].astype(int)
    df_info = df_info.reset_index(drop=True)
    df_temp = df_temp.reset_index(drop=True)
    df_info_final = pd.concat([df_info, df_temp], axis=1)
    return df_info_final


def df_info_inicial():
    #df = cargar_dataframe(url)
    df_info = df_disc.iloc[1:, :10]
    df_temp = df_disc.iloc[1:, -3]
    df_info.iloc[:, 0] = df_info.iloc[:, 0].map(int)
    df_info.rename(columns={'1': 'Id'}, inplace=True)
    df_info['Número de Cédula'] = df_info['Número de Cédula'].astype(int)
    df_info['Edad'] = df_info['Edad'].astype(int)
    df_info_total = pd.concat([df_info, df_temp], axis=1)
    #df_info = df_info.reset_index(drop=True)
    return df_info_total


def carga_inicial_disc(id):
    #df = cargar_dataframe(url)
    df_dicen = df_disc.iloc[[id]]
    df_dicen = df_dicen.iloc[:, 10:66]
    
    #df_dicen.to_excel("Respuestas Dicen.xlsx")
    df_pc = calcular_total_dicen(df_dicen)
    return  df_pc


def carga_care(id):
    #df = cargar_dataframe(url)
    df_care = df_disc.iloc[[id]]
    df_care = df_care.iloc[:, 66:114]
    
    #df_care.to_excel("Respuestas CARE.xlsx")
    df_pc_care = total_care(df_care)
    #df_perfil_care = perfil_care(df_pc_care)
    df_pc_care = df_pc_care.reset_index(drop=True)
    #df_perfil_care = df_perfil_care.reset_index(drop=True)
    #df_final_care = pd.concat([df_pc_care, df_perfil_care], axis=1)
    return df_pc_care


def carga_liderazgo(id):
    #df = cargar_dataframe(url)
    df_liderazgo = df_disc.iloc[[id]]
    df_liderazgo = df_liderazgo.iloc[:, 114:174]
    
    #df_liderazgo.to_excel("Respuestas liderazgo.xlsx")
    #df_liderazgo = df_liderazgo.reset_index(drop=True)
    df_result_liderazgo = liderazgo_total(df_liderazgo, id)
    df_result_liderazgo_total = liderazgo_orden(df_result_liderazgo)
    return df_result_liderazgo_total

def list_disc_for_graf(id):
    disc = carga_inicial_disc(id)
    total_ = disc.loc[:,['Dominante', 'Influyente', 'Concienzudo', 'Estable']].T
    disc_list_pc = total_[0].values.tolist()
    return disc_list_pc


def list_care_for_graf(id):
    df = carga_care(id)
    df = df.loc[:,['Conceptual', 'Espontáneo', 'Normativo', 'Metódico']].T
    list_care = df[0].values.tolist()
    return list_care


def list_lider_for_graf(id):
    df = carga_liderazgo(id)
    df = df.loc[:,['Entusiasmo', 'Integridad', 'Autorenovacion', 'Fortaleza', 'Percepción', 'Criterio', 'Ejecución', 'Audacia', 'Construcción de un equipo', 'Colaboración', 'Inspiración', 'Servir a los demás']].T
    list_lider = df[0].values.tolist()
    return list_lider


def carga_total_completo(id):
    info_total = info_test_total(id)
    dicen_total = carga_inicial_disc(id)
    care_total = carga_care(id)
    liderazgo_total = carga_liderazgo(id)
    
    #list_liderazgo_pd = liderazgo_orden(liderazgo_total)
    info_total = info_total.reset_index(drop=True)
    dicen_total = dicen_total.reset_index(drop=True)
    care_total = care_total.reset_index(drop=True)
    liderazgo_total = liderazgo_total.reset_index(drop=True)
    test_completo = pd.concat([info_total, dicen_total, care_total, liderazgo_total], axis=1)
    #test_completo.to_excel("Resultado DISC.xlsx", sheet_name="Perfil")
    return test_completo



def calcular_total_dicen(df):
    #df_dicen2 = df_dicen.copy(deep=True)
    column_mas = []
    column_menos = []
    for i in range(len(df.columns)):
        if i % 2 == 0:
            column_mas.append(i)
        else:
            column_menos.append(i)
    df_dicen_mas = df.iloc[:, column_mas]
    df_dicen_menos = df.iloc[:, column_menos]
    #Convierte las respuestas a numeros
    ##Positivos
    for i in range(len(df_dicen_mas.columns)):
        df_dicen_mas.iloc[:, i] = df_dicen_mas.iloc[:, i].apply(set_dicen)
    ##Negativos
    for i in range(len(df_dicen_menos.columns)):
        if(i==12):
            df_dicen_menos.iloc[:, i] = df_dicen_menos.iloc[:, i].apply(set_dicen3)
        else:
            df_dicen_menos.iloc[:, i] = df_dicen_menos.iloc[:, i].apply(set_dicen2)
    
    dic_total = {'Total D': df_dicen_mas[df_dicen_mas == 2].count(axis=1),
                 'Total I': df_dicen_mas[df_dicen_mas == 1].count(axis=1),
                 'Total C': df_dicen_mas[df_dicen_mas == 3].count(axis=1),
                 'Total E': df_dicen_mas[df_dicen_mas == 4].count(axis=1),
                 'Total N': df_dicen_mas[df_dicen_mas == 5].count(axis=1),
                 'Total D-': df_dicen_menos[df_dicen_menos == 2].count(axis=1),
                 'Total I-': df_dicen_menos[df_dicen_menos == 1].count(axis=1),
                 'Total C-': df_dicen_menos[df_dicen_menos == 3].count(axis=1),
                 'Total E-': df_dicen_menos[df_dicen_menos == 4].count(axis=1),
                 'Total N-': df_dicen_menos[df_dicen_menos == 6].count(axis=1),
                 'Total Dif D': df_dicen_mas[df_dicen_mas == 2].count(axis=1) - df_dicen_menos[df_dicen_menos == 2].count(axis=1),
                 'Total Dif I': df_dicen_mas[df_dicen_mas == 1].count(axis=1) - df_dicen_menos[df_dicen_menos == 1].count(axis=1),
                 'Total Dif C': df_dicen_mas[df_dicen_mas == 3].count(axis=1) - df_dicen_menos[df_dicen_menos == 3].count(axis=1),
                 'Total Dif E': df_dicen_mas[df_dicen_mas == 4].count(axis=1) - df_dicen_menos[df_dicen_menos == 4].count(axis=1),
                 'Total Dif N': df_dicen_mas[df_dicen_mas == 5].count(axis=1) - df_dicen_menos[df_dicen_menos == 6].count(axis=1),}

    df_dicen_total = pd.DataFrame(dic_total)

    pd_d = df_dicen_total.loc[:,'Total D'].values.tolist()
    pd_i = df_dicen_total.loc[:, 'Total I'].values.tolist()
    pd_c = df_dicen_total.loc[:, 'Total C'].values.tolist()
    pd_e = df_dicen_total.loc[:, 'Total E'].values.tolist()
    pd_n = df_dicen_total.loc[:, 'Total N'].values.tolist()
    pd_d_min = df_dicen_total.loc[:, 'Total D-'].values.tolist()
    pd_i_min = df_dicen_total.loc[:, 'Total I-'].values.tolist()
    pd_c_min = df_dicen_total.loc[:, 'Total C-'].values.tolist()
    pd_e_min = df_dicen_total.loc[:, 'Total E-'].values.tolist()
    pd_n_min = df_dicen_total.loc[:, 'Total N-'].values.tolist()
    pd_d_dif = df_dicen_total.loc[:, 'Total Dif D'].values.tolist()
    pd_i_dif = df_dicen_total.loc[:, 'Total Dif I'].values.tolist()
    pd_c_dif = df_dicen_total.loc[:, 'Total Dif C'].values.tolist()
    pd_e_dif = df_dicen_total.loc[:, 'Total Dif E'].values.tolist()
    #print(pd_d_min)
    dis_pc = {
        'PC Ds': percentil_dicen(pd_d_dif, 'C', 'PD', 'Pc d'),
        'PC I': percentil_dicen(pd_i_dif, 'C', 'PD', 'Pc i'),
        'PC C': percentil_dicen(pd_c_dif, 'C', 'PD', 'Pc c'),
        'PC E': percentil_dicen(pd_e_dif, 'C', 'PD', 'Pc s'),
        'PC MAS D': percentil_dicen(pd_d, 'A', 'Pc', 'D'),
        'PC MAS I': percentil_dicen(pd_i, 'A', 'Pc', 'I'),
        'PC MAS C': percentil_dicen(pd_c, 'A', 'Pc', 'C'),
        'PC MAS E': percentil_dicen(pd_e, 'A', 'Pc', 'S'),
        'PC MIN D': percentil_dicen(pd_d_min, 'B', 'Pc', 'D'),
        'PC MIN I': percentil_dicen(pd_i_min, 'B', 'Pc', 'I'),
        'PC MIN C': percentil_dicen(pd_c_min, 'B', 'Pc', 'C'),
        'PC MIN E': percentil_dicen(pd_e_min, 'B', 'Pc', 'S'),
        'PC DIF D': percentil_dicen(pd_d_dif, 'C', 'PD', 'D'),
        'PC DIF I': percentil_dicen(pd_i_dif, 'C', 'PD', 'I'),
        'PC DIF C': percentil_dicen(pd_c_dif, 'C', 'PD', 'C'),
        'PC DIF E': percentil_dicen(pd_e_dif, 'C', 'PD', 'S'),

    }
    df_pc_disc = pd.DataFrame(dis_pc)
    df_pc_disc['Dominante'] = round(df_pc_disc['PC Ds']*100/28).astype(int)
    df_pc_disc['Influyente'] = round(df_pc_disc['PC I']*100/28).astype(int)
    df_pc_disc['Concienzudo'] = round(df_pc_disc['PC C']*100/28).astype(int)
    df_pc_disc['Estable'] = round(df_pc_disc['PC E']*100/28).astype(int)
    df_pc_disc['MAS'] = df_pc_disc['PC MAS D'].astype(str) + df_pc_disc['PC MAS I'].astype(str) + df_pc_disc['PC MAS E'].astype(str) + df_pc_disc['PC MAS C'].astype(str)
    df_pc_disc['MENOS'] = df_pc_disc['PC MIN D'].astype(str) + df_pc_disc['PC MIN I'].astype(str) + df_pc_disc[
        'PC MIN E'].astype(str) + df_pc_disc['PC MIN C'].astype(str)
    df_pc_disc['DIFERENCIA'] = df_pc_disc['PC DIF D'].astype(str) + df_pc_disc['PC DIF I'].astype(str) + df_pc_disc[
        'PC DIF E'].astype(str) + df_pc_disc['PC DIF C'].astype(str)
    
    inicioapply = time.time()
    df_pc_disc['Patron MAS ALT'] = df_pc_disc['MAS'].apply(result_dicen_alternativo, baremo='patron', columna_comparar='Texto', columna_recuperar='Referencia')
    fin_apply = time.time()
    totalApply = fin_apply - inicioapply
    #print(totalApply)
    iniciolist = time.time()
    #df_pc_disc['Patron MAS list'] = df_pc_disc['MAS'].apply(result_dicen_alternativo, baremo='patron', columna_comparar='Texto', columna_recuperar='Referencia')
    mas_list = df_pc_disc['MAS'].values.tolist()
    df_patron_alt = {
        'Patron mas list': result_dicen(mas_list, 'patron', 'Texto', 'Referencia'),
    }
    fin_list = time.time()
    totalList = fin_list - iniciolist
    dominante_list = df_pc_disc['PC Ds'].values.tolist()
    influyente_list = df_pc_disc['PC I'].values.tolist()
    estable_list = df_pc_disc['PC E'].values.tolist()
    concienzudo_list = df_pc_disc['PC C'].values.tolist()
    inicio_listotal= time.time()
    df_patron = {
        'Patron mas': df_pc_disc['MAS'].apply(result_dicen_alternativo, baremo='patron', columna_comparar='Texto', columna_recuperar='Referencia'),
        'Patron menos': df_pc_disc['MENOS'].apply(result_dicen_alternativo,baremo='patron', columna_comparar='Texto', columna_recuperar='Referencia'),
        'Patron diferencia': df_pc_disc['DIFERENCIA'].apply(result_dicen_alternativo, baremo='patron', columna_comparar='Texto', columna_recuperar='Referencia'),
        'Dominante1': df_pc_disc['PC Ds'].apply(result_dicen_alternativo, baremo='dominante', columna_comparar='N°', columna_recuperar='Caracteristica'),
        'Dominante2': df_pc_disc['PC Ds'].apply(result_dicen_alternativo, baremo='dominante', columna_comparar='N°', columna_recuperar='Caracteristica3'),
        'Dominante3': df_pc_disc['PC Ds'].apply(result_dicen_alternativo, baremo='dominante', columna_comparar='N°', columna_recuperar='Caracteristica5'),
        'Influyente1': df_pc_disc['PC I'].apply(result_dicen_alternativo, baremo='influyente', columna_comparar='N°', columna_recuperar='Caracteristica'),
        'Influyente2': df_pc_disc['PC I'].apply(result_dicen_alternativo, baremo='influyente', columna_comparar='N°', columna_recuperar='Caracteristica3'),
        'Influyente3': df_pc_disc['PC I'].apply(result_dicen_alternativo, baremo='influyente', columna_comparar='N°', columna_recuperar='Caracteristica5'),
        'Estable1': df_pc_disc['PC E'].apply(result_dicen_alternativo, baremo='estable', columna_comparar='N°', columna_recuperar='Caracteristica'),
        'Estable2': df_pc_disc['PC E'].apply(result_dicen_alternativo, baremo='estable', columna_comparar='N°', columna_recuperar='Caracteristica3'),
        'Estable3': df_pc_disc['PC E'].apply(result_dicen_alternativo, baremo='estable', columna_comparar='N°', columna_recuperar='Caracteristica5'),
        'Concienzudo1': df_pc_disc['PC C'].apply(result_dicen_alternativo, baremo='concienzudo', columna_comparar='N°', columna_recuperar='Caracteristica'),
        'Concienzudo2': df_pc_disc['PC C'].apply(result_dicen_alternativo, baremo='concienzudo', columna_comparar='N°', columna_recuperar='Caracteristica3'),
        'Concienzudo3': df_pc_disc['PC C'].apply(result_dicen_alternativo, baremo='concienzudo', columna_comparar='N°', columna_recuperar='Caracteristica5'),

    }
    fin = time.time()
    total = fin - inicio_listotal
    df_max = df_pc_disc.loc[:,['Dominante', 'Influyente', 'Concienzudo', 'Estable']]
    df_max['MAX'] = df_max.idxmax(axis=1)
    
    df_max = df_max.reset_index(drop=True)
    df_max = df_max.loc[:,'MAX']
    df_max_list = df_max.values.tolist()
    df_det_disc = pd.DataFrame(df_patron)
    patron_disc = df_det_disc['Patron diferencia'].values.tolist()

    perfil_disc = {
        'Emociones': result_dicen(patron_disc, 'rol_enfoque_care', 'Perfil', 'Emociones'),
        'Meta': result_dicen(patron_disc, 'rol_enfoque_care', 'Perfil', 'Meta'),
        'Juzga': result_dicen(patron_disc, 'rol_enfoque_care', 'Perfil', 'Juzga'),
        'Influye': result_dicen(patron_disc, 'rol_enfoque_care', 'Perfil', 'Influye'),
        'Valor para la organización': result_dicen(patron_disc, 'rol_enfoque_care', 'Perfil', 'Valor para la organización'),
        'Añadido': result_dicen(patron_disc, 'rol_enfoque_care', 'Perfil', 'Añadido'),
        'Abusa': result_dicen(patron_disc, 'rol_enfoque_care', 'Perfil', 'Abusa'),
        'Bajo presión': result_dicen(patron_disc, 'rol_enfoque_care', 'Perfil', 'Bajo presión'),
        'Teme': result_dicen(patron_disc, 'rol_enfoque_care', 'Perfil', 'Teme'),
        'Observaciones': result_dicen(patron_disc, 'rol_enfoque_care', 'Perfil', 'Observaciones'),
        'Sugerencias': result_dicen(patron_disc, 'rol_enfoque_care', 'Perfil', 'Sugerencias'),
        'Enfasis': result_dicen(df_max_list, 'Perfil2', 'Perfil', 'Enfasis'),
        'Tendencias': result_dicen(df_max_list, 'Perfil2', 'Perfil', 'Tendencia'),
        'Necesidades de otros': result_dicen(df_max_list, 'Perfil2', 'Perfil', 'Necesidades de otros'),
        'Deseos': result_dicen(df_max_list, 'Perfil2', 'Perfil', 'Deseos'),
        'Aumento de eficacia': result_dicen(df_max_list, 'Perfil2', 'Perfil', 'Aumento de eficacia'),
        
        

    }
    df_perfil_disc = pd.DataFrame(perfil_disc)
    df_perfil_disc = df_perfil_disc.reset_index(drop=True)
    df_det_disc = df_det_disc.reset_index(drop=True)
    df_dicen_total = df_dicen_total.reset_index(drop=True)
    df_pc_disc = df_pc_disc.reset_index(drop=True)
    df_final = pd.concat([df_dicen_total, df_pc_disc, df_det_disc, df_max, df_perfil_disc], axis=1)
    return df_final
    # Se unen todos los dataframes
    #df_final = pd.concat([df_info, df_temp, df_dicen, df_dicen_total], axis=1)
    #print(df_final.iloc[:, -5:])


def total_care(df):
    totalpd = {
        'Conceptualx': df.iloc[:, 1] + df.iloc[:, 4] + df.iloc[:, 10] + df.iloc[:, 15] + df.iloc[:, 18] + df.iloc[:, 20] + df.iloc[:, 24] + df.iloc[:, 31] + df.iloc[:, 34] + df.iloc[:, 37] + df.iloc[:, 43] + df.iloc[:, 45],
        'Espontaneot': df.iloc[:, 2] + df.iloc[:, 7] + df.iloc[:, 8] + df.iloc[:, 12] + df.iloc[:, 17] + df.iloc[:, 23] + df.iloc[:, 27] + df.iloc[:, 29] + df.iloc[:, 33] + df.iloc[:, 38] + df.iloc[:, 40] + df.iloc[:, 47],
        'Normativor': df.iloc[:, 3] + df.iloc[:, 6] + df.iloc[:, 11] + df.iloc[:, 13] + df.iloc[:, 19] + df.iloc[:, 21] + df.iloc[:, 25] + df.iloc[:, 28] + df.iloc[:, 35] + df.iloc[:, 36] + df.iloc[:, 41] + df.iloc[:, 46],
        'Metodicoc': df.iloc[:, 0] + df.iloc[:, 5] + df.iloc[:, 9] + df.iloc[:, 14] + df.iloc[:, 16] + df.iloc[:, 22] + df.iloc[:, 26] + df.iloc[:, 30] + df.iloc[:, 32] + df.iloc[:, 39] + df.iloc[:, 42] + df.iloc[:, 44],

    }
    df_dimension = pd.DataFrame(totalpd)
    
    df_dimension['MAX_CARE'] = df_dimension.idxmax(axis=1)
    df_dimension['Total_CARE'] = df_dimension.iloc[:,:4].sum(axis=1).astype(int)
    result_x = df_dimension['Conceptualx'].values.tolist()
    result_t = df_dimension['Espontaneot'].values.tolist()
    result_r = df_dimension['Normativor'].values.tolist()
    result_c = df_dimension['Metodicoc'].values.tolist()
    total_care_result = {
        'Conceptual': result_dicen(result_x, 'tabla_care', 'X', 'Resultado x'),
        'Espontáneo': result_dicen(result_t, 'tabla_care', 'T', 'Resultado t'),
        'Normativo': result_dicen(result_r, 'tabla_care', 'R', 'Resultado r'),
        'Metódico': result_dicen(result_c, 'tabla_care', 'C', 'Resultado c'),

    }
    df_resultado = pd.DataFrame(total_care_result)

    df_dimension['Conceptualx'] = df_dimension['Conceptualx'].astype(int)
    df_dimension['Espontaneot'] = df_dimension['Espontaneot'].astype(int)
    df_dimension['Normativor'] = df_dimension['Normativor'].astype(int)
    df_dimension['Metodicoc'] = df_dimension['Metodicoc'].astype(int)

    df_resultado['Enfoque CARE'] = df_resultado.idxmax(axis=1)
    perfiles_care = {
        'Creador': (df_resultado.iloc[:, 0] * df_resultado.iloc[:, 1])/2,
        'Avanzador': (df_resultado.iloc[:, 1] * df_resultado.iloc[:, 2])/2,
        'Refinador': (df_resultado.iloc[:, 0] * df_resultado.iloc[:, 3])/2,
        'Ejecutor': (df_resultado.iloc[:, 2] * df_resultado.iloc[:, 3])/2,
    }
    df_perfil_care = pd.DataFrame(perfiles_care)
    df_perfil_care['Perfil CARE'] = df_perfil_care.idxmax(axis=1)
    
    #mask = df_perfil_care[]
    #print()
    df_resultado = df_resultado.reset_index(drop=True)
    df_dimension = df_dimension.reset_index(drop=True)
    df_perfil_care = df_perfil_care.reset_index(drop=True)
    

    #df = df.reset_index(drop=True)
    df_care = pd.concat([df_dimension, df_resultado, df_perfil_care], axis=1)
    # UNIR EL PERFIL CON LA DESCRIPCION
    df_desc_perfil_care = perfil_care(df_care)
    df_care = df_care.reset_index(drop=True)
    df_desc_perfil_care = df_desc_perfil_care.reset_index(drop=True)
    df_care1 = pd.concat([df_care, df_desc_perfil_care], axis=1)

    return df_care1

def perfil_care(df):
    df2 = pd.DataFrame()
    condiciones_rol = [ 
        ((df['Enfoque CARE'] == 'Espontáneo')&(df['Perfil CARE'] == 'Creador')) | 
        ((df['Enfoque CARE'] == 'Conceptual')&(df['Perfil CARE'] == 'Creador')),
        ((df['Enfoque CARE'] == 'Espontáneo')&(df['Perfil CARE'] == 'Avanzador')) |
        ((df['Enfoque CARE'] == 'Normativo')&(df['Perfil CARE'] == 'Avanzador')),
        ((df['Enfoque CARE'] == 'Conceptual')&(df['Perfil CARE'] == 'Refinador')) |
        ((df['Enfoque CARE'] == 'Metódico')&(df['Perfil CARE'] == 'Refinador')),
        ((df['Enfoque CARE'] == 'Metódico')&(df['Perfil CARE'] == 'Ejecutor')) |
        ((df['Enfoque CARE'] == 'Normativo')&(df['Perfil CARE'] == 'Ejecutor')),
        ((df['Enfoque CARE'] == 'Normativo')&(df['Perfil CARE'] == 'Creador')),
        ((df['Enfoque CARE'] == 'Conceptual')&(df['Perfil CARE'] == 'Avanzador')),
        ((df['Enfoque CARE'] == 'Metódico')&(df['Perfil CARE'] == 'Avanzador')),
        ((df['Enfoque CARE'] == 'Espontáneo')&(df['Perfil CARE'] == 'Ejecutor')),
        ((df['Enfoque CARE'] == 'Normativo')&(df['Perfil CARE'] == 'Refinador')),
        ((df['Enfoque CARE'] == 'Conceptual')&(df['Perfil CARE'] == 'Ejecutor')),
    ]
    seleccion_rol = ['Creador 1', 'Avanzador 1', 'Refinador 1', 'Ejecutor 1', 'Creador 2', 
                     'Avanzador 4', 'Avanzador 2', 'Ejecutor2', 'Refinador 2', 'Ejecutor 3' ]
    df2['Rol'] = np.select(condiciones_rol,seleccion_rol)

    rol_enfoque_list = df2['Rol'].values.tolist()
    descrip_enfoque_list = df['Enfoque CARE'].values.tolist()
    descrip_perfil_list = df['Perfil CARE'].values.tolist()
    descrip_care = {
        'Descripcion Enfoque': result_dicen(descrip_enfoque_list, 'enfoque_care', 'Enfoque', 'Descripcion'),
        'Descripcion Perfil': result_dicen(descrip_perfil_list, 'rol_care', 'Rol', 'Descripcion'),
        'Rol Desc': result_dicen(rol_enfoque_list, 'rol_enfoque_care_DESC', 'Busqueda', 'Rol'),
        'Descripción Rol': result_dicen(rol_enfoque_list, 'rol_enfoque_care_DESC', 'Busqueda', 'Descripción'),
        'Contribución Rol': result_dicen(rol_enfoque_list, 'rol_enfoque_care_DESC', 'Busqueda', 'Contribución'),
        'Satisfacción Rol': result_dicen(rol_enfoque_list, 'rol_enfoque_care_DESC', 'Busqueda', 'Satisfaccion'),
        'Debilidades Rol': result_dicen(rol_enfoque_list, 'rol_enfoque_care_DESC', 'Busqueda', 'Debilidades'),
        'Instinto Rol': result_dicen(rol_enfoque_list, 'rol_enfoque_care_DESC', 'Busqueda', 'Instinto'),
    }
    df_desc_care = pd.DataFrame(descrip_care)
    df2= df2.reset_index(drop=True)
    df_desc_care = df_desc_care.reset_index(drop=True)
    df_care_desc = pd.concat([df2, df_desc_care], axis=1)

    return df_care_desc
    
def liderazgo_total(df, id):
    totalpd = {
        'Entusiasmo': df.iloc[:, 6] + df.iloc[:, 20] + df.iloc[:, 28] + df.iloc[:, 36] + df.iloc[:, 51],
        'Integridad': df.iloc[:, 4] + df.iloc[:, 13] + df.iloc[:, 43] + df.iloc[:, 54] + df.iloc[:, 59] ,
        'Autorenovacion': df.iloc[:, 15] + df.iloc[:, 25] + df.iloc[:, 30] + df.iloc[:, 45] + df.iloc[:, 58] ,
        'Fortaleza': df.iloc[:, 2] + df.iloc[:, 11] + df.iloc[:, 38] + df.iloc[:, 50] + df.iloc[:, 57] ,
        'Percepción': df.iloc[:, 1] + df.iloc[:, 23] + df.iloc[:, 26] + df.iloc[:, 32] + df.iloc[:, 47] ,
        'Criterio': df.iloc[:, 8] + df.iloc[:, 10] + df.iloc[:, 16] + df.iloc[:, 21] + df.iloc[:, 40] ,
        'Ejecución': df.iloc[:, 9] + df.iloc[:, 27] + df.iloc[:, 39] + df.iloc[:, 42] + df.iloc[:, 48] ,
        'Audacia': df.iloc[:, 0] + df.iloc[:, 17] + df.iloc[:, 24] + df.iloc[:, 33] + df.iloc[:, 52] ,
        'Construcción de un equipo': df.iloc[:, 14] + df.iloc[:, 19] + df.iloc[:, 41] + df.iloc[:, 49] + df.iloc[:, 56] ,
        'Colaboración': df.iloc[:, 5] + df.iloc[:, 18] + df.iloc[:, 22] + df.iloc[:, 34] + df.iloc[:, 53] ,
        'Inspiración': df.iloc[:, 31] + df.iloc[:, 37] + df.iloc[:, 44] + df.iloc[:, 46] + df.iloc[:, 55] ,
        'Servir a los demás': df.iloc[:, 3] + df.iloc[:, 7] + df.iloc[:, 12] + df.iloc[:, 29] + df.iloc[:, 35] ,

    }

    df_pd_liderazgo = pd.DataFrame(totalpd)
    df_pd_liderazgo = df_pd_liderazgo[df_pd_liderazgo.columns].fillna(0).astype(int)
    
    
    """ df_pd_liderazgo['Dimensión mayor Liderazgo'] = df_pd_liderazgo.idxmax(axis=1)
    df_pd_liderazgo['Dimensión menor Liderazgo'] = df_pd_liderazgo.iloc[:,:-1].idxmin(axis=1)
    df_pd_liderazgo['Max Dimension Liderazgo'] = df_pd_liderazgo.iloc[:,:-2].max(axis=1)
    df_pd_liderazgo['Min Dimension Liderazgo'] = df_pd_liderazgo.iloc[:,:-3].min(axis=1)

    df_pd_liderazgo['Criterio mayor Liderazgo'] = np.where(df_pd_liderazgo['Max Dimension Liderazgo']<20,'No', 'Si')
    df_pd_liderazgo['Criterio menor Liderazgo'] = np.where(df_pd_liderazgo['Min Dimension Liderazgo']<11,'No', 'Si')
    df_ pd_liderazgo['Criterio segundo menor Liderazgo'] = np.where(df_pd_liderazgo['Min Dimension Liderazgo']<11,'No', 'Si')"""
    total_pd_enfoque = {
        'Carácter': df_pd_liderazgo.iloc[:, 0] + df_pd_liderazgo.iloc[:, 1] + df_pd_liderazgo.iloc[:, 2],
        'Análisis': df_pd_liderazgo.iloc[:, 3] + df_pd_liderazgo.iloc[:, 4] + df_pd_liderazgo.iloc[:, 5],
        'Realización': df_pd_liderazgo.iloc[:, 6] + df_pd_liderazgo.iloc[:, 7] + df_pd_liderazgo.iloc[:, 8],
        'Interacción': df_pd_liderazgo.iloc[:, 9] + df_pd_liderazgo.iloc[:, 10] + df_pd_liderazgo.iloc[:, 11],
    }
    df_total_enfoque = pd.DataFrame(total_pd_enfoque)
    df_total_enfoque = df_total_enfoque[df_total_enfoque.columns].fillna(0).astype(int)
    """ print(df_pd_liderazgo)
    print(df_total_enfoque) """
    #print(df_pd_liderazgo.loc[:, ['Min Dimension Liderazgo', 'Dimensión menor Liderazgo', 'Criterio menor Liderazgo']])
    
    #id = str(id)
    df_pd_liderazgo= df_pd_liderazgo.reset_index(drop=True)
    df_total_enfoque = df_total_enfoque.reset_index(drop=True)
    #print(df_pd_liderazgo)
    df_pd_liderazgo = df_pd_liderazgo.sort_values(by=0, axis=1)
    df_total_enfoque = df_total_enfoque.sort_values(by=0, axis=1)
    
    df_pd_liderazgo_corregido = pd.concat([df_pd_liderazgo, df_total_enfoque], axis=1)
    return df_pd_liderazgo_corregido


def liderazgo_orden(df):
    row_liderazgo= df.iloc[:, :-4]
    row_enfoque = df.iloc[:,-4:]
    """ row_liderazgo = df_total.iloc[[id]]
    row_enfoque = df_enfoque.iloc[[id]]
    
    row_liderazgo = row_liderazgo.sort_values(by=id, axis=1)
    
    row_enfoque = row_enfoque.sort_values(by=id, axis=1) """
    
    liderazgo_max_min = {
        'Max Liderazgo 1': [row_liderazgo.columns[-1]],
        'Max Liderazgo 2': [row_liderazgo.columns[-2]],
        'Max Liderazgo 3': [row_liderazgo.columns[-3]],
        'Min Liderazgo 1': [row_liderazgo.columns[0]],
        'Min Liderazgo 2': [row_liderazgo.columns[1]],
        'Min Liderazgo 3': [row_liderazgo.columns[2]],
    }
    enfoque_max_min = {
        'Max Enfoque 1': [row_enfoque.columns[-1]],
        'Max Enfoque 2': [row_enfoque.columns[-2]],
    }
    max_min_liderazgo = pd.DataFrame(liderazgo_max_min)
    max_min_enfoque = pd.DataFrame(enfoque_max_min)

    max_min_liderazgo['Criterio mayor Liderazgo 1'] = np.where(row_liderazgo.iloc[:, -1]<20,'No', 'Si')
    max_min_liderazgo['Criterio mayor Liderazgo 2'] = np.where(row_liderazgo.iloc[:, -2]<20,'No', 'Si')
    max_min_liderazgo['Criterio mayor Liderazgo 3'] = np.where(row_liderazgo.iloc[:, -3]<20,'No', 'Si')
    max_min_liderazgo['Criterio menor Liderazgo 1'] = np.where(row_liderazgo.iloc[:, 0]<11,'Si', 'No')
    max_min_liderazgo['Criterio menor Liderazgo 2'] = np.where(row_liderazgo.iloc[:, 1]<11,'Si', 'No')
    max_min_liderazgo['Criterio menor Liderazgo 3'] = np.where(row_liderazgo.iloc[:, 2]<11,'Si', 'No')
    max_min_enfoque['Criterio Enfoque 2'] = np.where(row_enfoque.iloc[:, -1]==row_enfoque.iloc[:, -2],'Si', 'No')
    describ_nombre_lider_list = max_min_enfoque['Max Enfoque 1'].values.tolist()
    describ_nombre_lider_list2 = max_min_enfoque['Max Enfoque 2'].values.tolist()
    describ_dimension_lider_list_max = max_min_liderazgo['Max Liderazgo 1'].values.tolist()
    describ_dimension_lider_list_max2 = max_min_liderazgo['Max Liderazgo 2'].values.tolist()
    describ_dimension_lider_list_max3 = max_min_liderazgo['Max Liderazgo 3'].values.tolist()
    describ_dimension_lider_list_min = max_min_liderazgo['Min Liderazgo 1'].values.tolist()
    describ_dimension_lider_list_min2 = max_min_liderazgo['Min Liderazgo 2'].values.tolist()

    describ_liderazgo = {
        'Nombre Enfoque liderazgo': result_dicen(describ_nombre_lider_list, 'liderazgo_enfoque', 'Enfoque', 'Nombre enfoque'),
        'Nombre Enfoque liderazgo2': result_dicen(describ_nombre_lider_list2, 'liderazgo_enfoque', 'Enfoque', 'Nombre enfoque'),
        'Desc Gral Enfoque liderazgo': result_dicen(describ_nombre_lider_list, 'liderazgo_enfoque', 'Enfoque', 'Desc general'),
        'Desc Gral Enfoque liderazgo2': result_dicen(describ_nombre_lider_list2, 'liderazgo_enfoque', 'Enfoque', 'Desc general'),
        'Desc Part Enfoque liderazgo': result_dicen(describ_nombre_lider_list, 'liderazgo_enfoque', 'Enfoque', 'Desc particular'),
        'Desc Part Enfoque liderazgo2': result_dicen(describ_nombre_lider_list2, 'liderazgo_enfoque', 'Enfoque', 'Desc particular'),
        'Desc Dimension alta liderazgo': result_dicen(describ_dimension_lider_list_max, 'liderazgo_dimension', 'Dimension', 'Descripcion1'),
        'Desc Dimension alta liderazgo2': result_dicen(describ_dimension_lider_list_max2, 'liderazgo_dimension', 'Dimension', 'Descripcion1'),
        'Desc Dimension alta liderazgo3': result_dicen(describ_dimension_lider_list_max3, 'liderazgo_dimension', 'Dimension', 'Descripcion1'),
        'Desc Dimension baja liderazgo': result_dicen(describ_dimension_lider_list_min, 'liderazgo_dimension', 'Dimension', 'Descripcion1'),
        'Desc Dimension baja liderazgo2': result_dicen(describ_dimension_lider_list_min2, 'liderazgo_dimension', 'Dimension', 'Descripcion1'),
        'Desc alto enfasis liderazgo': result_dicen(describ_dimension_lider_list_max, 'liderazgo_dimension', 'Dimension', 'Alto enfasis'),
        'Desc alto enfasis liderazgo2': result_dicen(describ_dimension_lider_list_max2, 'liderazgo_dimension', 'Dimension', 'Alto enfasis2'),
        'Desc alto enfasis liderazgo3': result_dicen(describ_dimension_lider_list_max3, 'liderazgo_dimension', 'Dimension', 'Alto enfasis3'),
        'Desc bajo enfasis liderazgo': result_dicen(describ_dimension_lider_list_min, 'liderazgo_dimension', 'Dimension', 'Bajo enfasis'),
        'Desc bajo enfasis liderazgo2': result_dicen(describ_dimension_lider_list_min2, 'liderazgo_dimension', 'Dimension', 'Bajo enfasis 2'),
        'Razon de seguir liderazgo': result_dicen(describ_dimension_lider_list_max, 'liderazgo_dimension', 'Dimension', 'Razon de seguimiento'),
        'Seguido por liderazgo': result_dicen(describ_dimension_lider_list_max, 'liderazgo_dimension', 'Dimension', 'Seguido por'),
        'Cuando guiar liderazgo': result_dicen(describ_dimension_lider_list_max, 'liderazgo_dimension', 'Dimension', 'Cuando guiar'),
        'Cuidados liderazgo': result_dicen(describ_dimension_lider_list_max, 'liderazgo_dimension', 'Dimension', 'Cuidados'),

    }
    df_perfil_liderazgo = pd.DataFrame(describ_liderazgo)
    row_liderazgo = row_liderazgo.reset_index(drop=True)
    max_min_liderazgo = max_min_liderazgo.reset_index(drop=True)
    row_enfoque = row_enfoque.reset_index(drop=True)
    max_min_enfoque = max_min_enfoque.reset_index(drop=True)
    df_enfoque_perfil = pd.concat([row_enfoque, max_min_enfoque, row_liderazgo, max_min_liderazgo, df_perfil_liderazgo],axis=1)

    #df_enfoque_perfil.to_excel("liderazgo_perfil.xlsx", sheet_name="Perfil liderazgo")
    #print(max_min_liderazgo)
    return df_enfoque_perfil
    
"""df = carga_care(url)
print(df.loc[:, ['Enfoque CARE', 'Perfil CARE', 'Rol', 'Instinto Rol']])
 df_car = perfil_care(df)
print(df_car.loc[:, ['Enfoque CARE', 'Perfil CARE', 'Instinto Rol']])
 

gr = get_disc_graf(3)

print(gr)"""
#df.to_excel("salida.xlsx")
# result=df.apply(lambda x: x.value_counts()).fillna(0)
#dz = info_test_total(url)
#print(dz.loc[:, ['Nombre y apellido', 'Número de Cédula', 'Fecha']])
#print(dz.iloc[3,:2])
#dg = carga_total_completo(3)