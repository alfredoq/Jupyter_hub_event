from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
from IPython.display import display
import py3Dmol
import ipywidgets as widgets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import linregress

def construct_molecule_2d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol

def MolTo3DView(smiles, size=(300, 300), style="stick", surface=False, opacity=0.5):
    mol = Chem.MolFromSmiles(smiles)
    mol_opt = optimize_conf(mol)
    mblock = Chem.MolToMolBlock(mol_opt)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({style:{}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer

def optimize_conf(mol):
    if mol is not None:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        return mol
    return None

def hydrolysis_products(smiles):
    mol = Chem.MolFromSmiles(smiles)
    hydrolysis_products_list = []

    ester_pattern = Chem.MolFromSmarts('[CX3:1](=[O:2])[OX2:3][C:4]')
    amide_pattern = Chem.MolFromSmarts('[CX3:1](=[O:2])[NX3,NX2,#7]')

    if mol.HasSubstructMatch(ester_pattern):
        if not mol.HasSubstructMatch(amide_pattern):
            print('Posee un éster!')
            rxn_smarts = '[CX3:1](=[O:2])[OX2:3][C:4].[OX2;H2:5]>>[CX3:1](=[O:2])[OX2;H1:3].[C:4][OX2;H1:5]'
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            products = rxn.RunReactants((mol, Chem.MolFromSmiles('O')))
            for product in products:
                for p in product:
                    product_smiles = Chem.MolToSmiles(p)
                    hydrolysis_products_list.append(product_smiles)

    if mol.HasSubstructMatch(amide_pattern) and not mol.HasSubstructMatch(ester_pattern):
        print('Posee una amida!')
        rxn_smarts_1 = '[CX3:1](=[O:2])[NX3,NX2,#7;H1:3].[OX2;H2:5]>>[CX3:1](=[O:2])[OX2;H1:5].[NX3,NX2,#7;H2:3]'
        rxn_smarts_2 = '[CX3:1](=[O:2])[NX3,NX2,#7:3].[OX2;H2:5]>>[CX3:1](=[O:2])[OX2;H1:5].[NX3,NX2,#7;H1:3]'
        try:
            rxn = AllChem.ReactionFromSmarts(rxn_smarts_1)
            products = rxn.RunReactants((mol, Chem.MolFromSmiles('O')))

            if not products:
                rxn = AllChem.ReactionFromSmarts(rxn_smarts_2)
                products = rxn.RunReactants((mol, Chem.MolFromSmiles('O')))

            for product in products:
                for p in product:
                    product_smiles = Chem.MolToSmiles(p)
                    hydrolysis_products_list.append(product_smiles)
        except Exception as e:
            print(f"Error al procesar la reacción: {e}")

    if mol.HasSubstructMatch(ester_pattern) and mol.HasSubstructMatch(amide_pattern):
        print('Posee tanto éster como amida!')
        rxn_smarts = '[CX3:1](=[O:2])[OX2:3][C:4].[OX2;H2:5]>>[CX3:1](=[O:2])[OX2;H1:3].[C:4][OX2;H1:5]'
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        products = rxn.RunReactants((mol, Chem.MolFromSmiles('O')))
        for product in products:
            for p in product:
                product_smiles = Chem.MolToSmiles(p)
                hydrolysis_products_list.append(product_smiles)
        for product_smiles in hydrolysis_products_list:
            product_mol = Chem.MolFromSmiles(product_smiles)
            rxn_smarts_amide = '[CX3:1](=[O:2])[NX3;H1:3].[OX2;H2:5]>>[CX3:1](=[O:2])[OX2;H1:5].[NX3;H2:3]'
            rxn_amide = AllChem.ReactionFromSmarts(rxn_smarts_amide)
            products_amide = rxn_amide.RunReactants((product_mol, Chem.MolFromSmiles('O')))
            for product_amide in products_amide:
                for p_amide in product_amide:
                    product_amide_smiles = Chem.MolToSmiles(p_amide)
                    hydrolysis_products_list.append(product_amide_smiles)
    

    
    hydrolysis_products_list = list(set(hydrolysis_products_list))  # Eliminar duplicados

    #print("Reactivo y Producto(s) en 2D:")
    mols = [construct_molecule_2d(smiles)] + [construct_molecule_2d(p) for p in hydrolysis_products_list]
    img = Draw.MolsToGridImage(mols, molsPerRow=len(mols), subImgSize=(300, 300), legends=["Reactivo"] + ["Producto"] * len(hydrolysis_products_list))
    display(img)

    #print("\nVisualización en 3D:")

    # Crear visualizadores 3D para el reactivo y los productos
    reactivo_3D = MolTo3DView(smiles, surface=False)
    productos_3D = [MolTo3DView(p, surface=False) for p in hydrolysis_products_list]

    # Crear widgets para mostrar los visores en una misma fila
    reactivo_widget = widgets.Output()
    with reactivo_widget:
        reactivo_3D.show()  # Sin usar display() para evitar "None"

    productos_widgets = []
    for producto in productos_3D:
        producto_widget = widgets.Output()
        with producto_widget:
            producto.show()  # Sin usar display() para evitar "None"
        productos_widgets.append(producto_widget)

    # Mostrar los visores alineados horizontalmente
    display(widgets.HBox([reactivo_widget] + productos_widgets))

def oxidize_molecule(input_smiles):
    mol = Chem.MolFromSmiles(input_smiles)

    # Definir patrones de catecol
    catecol_pattern1 = Chem.MolFromSmarts('[CX3,#6:1]([OH:2])=[#6:3]([OH:4])')
    catecol_pattern2 = Chem.MolFromSmarts('[CX3,#6:1]([OH:2])[#6:3]([OH:4])')

    # Inicializar conjuntos de productos
    unique_products = set()

    if mol.HasSubstructMatch(catecol_pattern1) or mol.HasSubstructMatch(catecol_pattern2):
        # Intentar con la primera reacción SMARTS
        rxn_smarts1 = '[CX3,#6:1]([OH:2])=[CX3,#6:3]([OH:4])>>[CX3,#6:1](=[O:2])[CX3,#6:3](=[O:4])'
        rxn1 = AllChem.ReactionFromSmarts(rxn_smarts1)
        products1 = rxn1.RunReactants((mol,))
        for product1 in products1:
            for p1 in product1:
                unique_products.add(Chem.MolToSmiles(p1))

        # Si no hubo productos, intentar con la segunda reacción
        if not unique_products:
            rxn_smarts2 = '[CX3,#6:1]([OH:2])[CX3,#6:3]([OH:4])>>[CX3,#6:1](=[O:2])[CX3,#6:3](=[O:4])'
            rxn2 = AllChem.ReactionFromSmarts(rxn_smarts2)
            products2 = rxn2.RunReactants((mol,))
            for product2 in products2:
                for p2 in product2:
                    unique_products.add(Chem.MolToSmiles(p2))
    if not unique_products:
        print("No tiene grupos oxidables.")
        
    # Crear lista de moléculas para visualización 2D
    mols_2d = [Chem.MolFromSmiles(input_smiles)] + [Chem.MolFromSmiles(p) for p in unique_products]

    # Mostrar imágenes 2D alineadas
    #print("Reactivo y Producto(s) en 2D:")
    img = Draw.MolsToGridImage(mols_2d, molsPerRow=len(mols_2d), subImgSize=(300, 300), 
                               legends=["Reactivo"] + ["Producto"] * len(unique_products))
    display(img)

    # Crear visualizadores 3D para el reactivo y los productos
    #print("\nVisualización en 3D:")
    reactivo_3D = MolTo3DView(input_smiles, surface=False)
    productos_3D = [MolTo3DView(p, surface=False) for p in unique_products]

    # Crear widgets de visualización 3D alineados
    reactivo_widget = widgets.Output()
    with reactivo_widget:
        reactivo_3D.show()

    productos_widgets = []
    for producto in productos_3D:
        producto_widget = widgets.Output()
        with producto_widget:
            producto.show()
        productos_widgets.append(producto_widget)

    # Mostrar las visualizaciones 3D alineadas horizontalmente
    display(widgets.HBox([reactivo_widget] + productos_widgets))
    
def calcular_parametros_cineticos(orden_reaccion, k=None, c0=None, t50=None, t90=None):
    """
    Calcula parámetros cinéticos como constantes de velocidad (k), concentraciones iniciales (c0),
    tiempos de vida media (t50) y tiempos de vida útil (t90) para reacciones de orden 0, 1 y 2.

    Parámetros:
    - orden_reaccion: int (0, 1 o 2)
    - k: constante de velocidad (opcional si se desea calcular)
    - c0: concentración inicial (opcional si se desea calcular)
    - t50: tiempo de vida media (opcional si se desea calcular)
    - t90: tiempo de vida útil (opcional si se desea calcular)

    Retorna:
    - k, c0, t50, t90 (con los valores calculados o los originales si no fueron calculados)
    """

    # Verificar cuántos valores son None
    parametros = {"k": k, "c0": c0, "t50": t50, "t90": t90}
    faltantes = [param for param, value in parametros.items() if value is None]

    if len(faltantes) > 2:
        raise ValueError("Debe proporcionarse al menos un valor entre k, t50 o t90.")

    # Cálculos según el orden de la reacción
    if orden_reaccion == 1:  # Reacción de primer orden
        c0 = None
        if k is not None:
            if t50 is None:
                t50 = np.log(2) / k
            if t90 is None:
                t90 = np.log(0.9) * -1 / k
        elif t50 is not None:
            k = np.log(2) / t50
            t90 = np.log(0.9) * -1 / k
        elif t90 is not None:
            k = np.log(0.9) * -1 / t90
            t50 = np.log(2) / k

    elif orden_reaccion == 2:  # Reacción de segundo orden
        if c0 is None:
            raise ValueError("Para una reacción de segundo orden, se debe proporcionar c0.")

        if k is not None:
            if t50 is None:
                t50 = 1 / (k * c0)
            if t90 is None:
                t90 = ((1/0.9) - 1) / (k * c0)
        elif t50 is not None:
            k = 1 / (t50 * c0)
            t90 = ((1/0.9) - 1) / (k * c0)
        elif t90 is not None:
            k = ((1/0.9) - 1) / (t90 * c0)
            t50 = 1 / (k * c0)

    elif orden_reaccion == 0:  # Reacción de orden cero
        if c0 is None:
            raise ValueError("Para una reacción de orden cero, se debe proporcionar c0.")

        if k is not None:
            if t50 is None:
                t50 = c0 / (2 * k)
            if t90 is None:
                t90 = (c0 * 0.1) / k
        elif t50 is not None:
            k = c0 / (t50 * 2)
            t90 = (c0 * 0.1) / k
        elif t90 is not None:
            k = (c0 * 0.1) / t90
            t50 = c0 / (2 * k)
    # Redondear los valores a 2 decimales antes de devolverlos
    k = "{:.2e}".format(k) if k is not None else None
    c0 = round(c0, 2) if c0 is not None else None
    t50 = round(t50, 2) if t50 is not None else None
    t90 = round(t90, 2) if t90 is not None else None
    
    print("¡No se olviden de agregar unidades!")
    
    return k, c0, t50, t90


def calcular_parametros_arrhenius(T1, T2, k1=None, k2=None, Ea=None, R=1.987):
    """
    Calcula la energía de activación (Ea) o la constante de velocidad (k1 o k2) 
    usando la ecuación de Arrhenius.

    Parámetros:
    - T1: temperatura 1 (en Kelvin)
    - T2: temperatura 2 (en Kelvin)
    - k1: constante de velocidad a T1 (opcional si se desea calcular)
    - k2: constante de velocidad a T2 (opcional si se desea calcular)
    - Ea: energía de activación (opcional si se desea calcular)
    - R: constante de los gases (1.987 cal/mol·K por defecto)

    Retorna:
    - Ea (en kJ/mol con 2 decimales) si se calcula Ea
    - k1 o k2 (en notación científica) si se calcula una constante de velocidad
    """

    # Contar cuántos valores son None
    parametros = {"k1": k1, "k2": k2, "Ea": Ea}
    faltantes = [param for param, value in parametros.items() if value is None]

    if len(faltantes) != 1:
        raise ValueError("Debe especificar exactamente una incógnita (k1, k2 o Ea).")

    # Calcular Ea si falta
    if Ea is None:
        Ea = R * np.log(k2 / k1) / (1 / T1 - 1 / T2)
        Ea = round(Ea, 2)  
        print("¡Recuerden que este resultado está en cal/mol!")
        return Ea

    # Calcular k2 si falta
    elif k2 is None:
        k2 = k1 * np.exp(- (Ea / R) * ((1 / T2) - (1 / T1)))
        k2 = "{:.2e}".format(k2)  # Notación científica
        print("¡Recuerden las unidades de k!")
        return k2

    # Calcular k1 si falta
    elif k1 is None:
        k1 = k2 / np.exp(- (Ea / R) * ((1 / T2) - (1 / T1)))
        k1 = "{:.2e}".format(k1)  # Notación científica
        print("¡Recuerden las unidades de k!")
        return k1

    
def calcular_concentracion(absorbancia, coef_extincion=1, longitud_camino=1):
    concentracion = absorbancia / (longitud_camino * coef_extincion)
    return concentracion

def generar_dataframe_desde_Abs(df, coef_extincion=1, longitud_camino=1):
    if 'concentracion' not in df.columns:
        df['concentracion'] = calcular_concentracion(df['absorbancia'], coef_extincion, longitud_camino)
        df['ln_concentracion'] = np.log(df['concentracion'])
        df['inversa_concentracion'] = 1 / df['concentracion']
    return df

def cinetica_orden_cero(t, c0, k):
    return c0 - k * t

def ajustar_cinetica_orden_cero(tiempo, concentracion):
    parametros_iniciales = [concentracion.iloc[0], 0.001]
    parametros_optimizados, matriz_covarianza = curve_fit(cinetica_orden_cero, tiempo, concentracion, p0=parametros_iniciales)

    concentracion_ajustada = cinetica_orden_cero(tiempo, *parametros_optimizados)
    r2 = r2_score(concentracion, concentracion_ajustada)

    return parametros_optimizados[0], parametros_optimizados[1], r2#, sse

def cinetica_orden_uno(t, c0, k):
    return c0 - k * t

def ajustar_cinetica_orden_uno(tiempo, concentracion_ln):
    slope, intercept, r_value, p_value, std_err = linregress(tiempo, concentracion_ln)
    r2 = r_value**2

    return intercept, -slope, r2#, sse

def cinetica_orden_dos(t, c0, k):
    return (1 / c0) + k * t

def ajustar_cinetica_orden_dos(tiempo, inversa_concentracion):
    parametros_iniciales = [1 / inversa_concentracion.iloc[0], 0.001]
    parametros_optimizados, matriz_covarianza = curve_fit(cinetica_orden_dos, tiempo, inversa_concentracion, p0=parametros_iniciales)

    concentracion_ajustada = cinetica_orden_dos(tiempo, *parametros_optimizados)
    r2 = r2_score(inversa_concentracion, concentracion_ajustada)

    return parametros_optimizados[0], parametros_optimizados[1], r2#, sse

def graficar_cineticas(tiempo, datos, parametros_optimizados, orden, color_puntos='blue', color_linea='red'):
    tiempo = np.array(tiempo)
    datos = np.array(datos)

    plt.subplot(1, 3, orden)
    sns.scatterplot(x=tiempo, y=datos, label='Datos experimentales', color=color_puntos)
    plt.xlabel('Tiempo')

    if orden == 1:
        plt.ylabel('Concentración')
        plt.plot(tiempo, cinetica_orden_cero(tiempo, *parametros_optimizados[:2]), label='Ajuste de orden cero', color=color_linea)
        plt.legend().set_visible(False)
        plt.title('Orden 0')

    elif orden == 2:
        plt.ylabel('ln(Concentración)')
        plt.plot(tiempo, cinetica_orden_uno(tiempo, *parametros_optimizados[:2]), label='Ajuste de orden uno', color=color_linea)
        plt.legend().set_visible(False)
        plt.title('Orden 1')

    elif orden == 3:
        plt.ylabel('1 / Concentración')
        plt.plot(tiempo, cinetica_orden_dos(tiempo, *parametros_optimizados[:2]), label='Ajuste de orden dos', color=color_linea)
        plt.legend().set_visible(False)
        plt.title('Orden 2')

    k_cientifico = '{:.2e}'.format(parametros_optimizados[1])

    if orden == 1 or orden == 2:
        texto = f'k: {k_cientifico}\nOrdenada al origen: {parametros_optimizados[0]:.4f}\n$r^2$: {parametros_optimizados[2]:.4f}'
    else:
        texto = f'k: {k_cientifico}\nOrdenada al origen: {(1/parametros_optimizados[0]):.4f}\n$r^2$:{parametros_optimizados[2]:.4f}'

    plt.annotate(texto, xy=(0.3, 0.85), xycoords='axes fraction', fontsize=10, color='black')
    
def analizar_cineticas(df, coef_extincion=1, longitud_camino=1):
    df = generar_dataframe_desde_Abs(df, coef_extincion, longitud_camino)

    # Inicializar listas para almacenar métricas de cada orden
    r2_values = []

    # Orden 0
    constante_ajuste_0, ordenada_al_origen_ajuste_0, r2_ajuste_0 = ajustar_cinetica_orden_cero(df['tiempo'], df['concentracion'])
    r2_values.append(r2_ajuste_0)
    print("Orden 0:")
    print(f"Ordenada al origen de ajuste: {round(constante_ajuste_0,5)}")
    print(f"Constante de ajuste: {round(ordenada_al_origen_ajuste_0,10)}")
    print(f"Coeficiente de determinación (r²): {round(r2_ajuste_0,5)}")

    # Orden 1
    constante_ajuste_1, ordenada_al_origen_ajuste_1, r2_ajuste_1 = ajustar_cinetica_orden_uno(df['tiempo'], np.log(df['concentracion']))
    r2_values.append(r2_ajuste_1)
    print("\nOrden 1:")
    print(f"Ordenada al origen de ajuste: {round(constante_ajuste_1,5)}")
    print(f"Constante de ajuste: {round(ordenada_al_origen_ajuste_1,10)}")
    print(f"Coeficiente de determinación (r²): {round(r2_ajuste_1,5)}")

    # Orden 2
    constante_ajuste_2, ordenada_al_origen_ajuste_2, r2_ajuste_2 = ajustar_cinetica_orden_dos(df['tiempo'], 1 / df['concentracion'])
    r2_values.append(r2_ajuste_2)
    print("\nOrden 2:")
    print(f"Ordenada al origen de ajuste: {round(1/constante_ajuste_2,5)}")
    print(f"Constante de ajuste: {round(ordenada_al_origen_ajuste_2,10)}")    
    print(f"Coeficiente de determinación (r²): {round(r2_ajuste_2,5)}")

    # Gráficos
    plt.figure(figsize=(15, 5))

    graficar_cineticas(df['tiempo'], df['concentracion'], [constante_ajuste_0, ordenada_al_origen_ajuste_0, r2_ajuste_0], 1, color_puntos='blue', color_linea='blue')
    graficar_cineticas(df['tiempo'], np.log(df['concentracion']), [constante_ajuste_1, ordenada_al_origen_ajuste_1, r2_ajuste_1], 2, color_puntos='green', color_linea='green')
    graficar_cineticas(df['tiempo'], 1 / df['concentracion'], [constante_ajuste_2, ordenada_al_origen_ajuste_2, r2_ajuste_2], 3, color_puntos='red', color_linea='red')

    plt.tight_layout()
    plt.show()

    # Comparación de r2 y determinación del orden
    r2_values = [r2_ajuste_0, r2_ajuste_1, r2_ajuste_2]
    ordenes = ['Orden 0', 'Orden 1', 'Orden 2']
    mejor_orden = ordenes[np.argmax(r2_values)]

     # Determinar el mejor orden según r2 y SSE
    mejor_orden_r2 = np.argmax(r2_values)

    print("\nResultados según r²:")
    print(f"Orden 0: {round(r2_values[0],5)}")
    print(f"Orden 1: {round(r2_values[1],5)}")
    print(f"Orden 2: {round(r2_values[2],5)}")

    print(f"\nMejor orden según r²: Orden {mejor_orden_r2}")

    
def preparar_dataframe(arrhenius_data, temp_unit='C'):
    # Crear DataFrame original
    arrhenius_data_df = pd.DataFrame(arrhenius_data)

    # Convertir temperatura a Kelvin si es necesario
    if temp_unit == 'C':
        arrhenius_data_df['Temperature_K'] = arrhenius_data_df['Temperature'] + 273.15
    elif temp_unit == 'K':
        arrhenius_data_df['Temperature_K'] = arrhenius_data_df['Temperature']

    # Agregar columnas al nuevo DataFrame
    arrhenius_data_df['ln_k'] = np.log(arrhenius_data_df['k'])
    arrhenius_data_df['1_over_T'] = 1 / arrhenius_data_df['Temperature_K']

    return arrhenius_data_df

def arrhenius_equation(T, A, Ea_over_R):
    return A * np.exp(-Ea_over_R / T)

def linealizar_arrhenius(T, k):
    # Linealizar ln(k) = -Ea/R * 1/T + ln(A)
    inv_T = 1 / T
    ln_k = np.log(k)

    # Ajuste lineal
    slope, intercept, r_value, p_value, std_err = linregress(inv_T, ln_k)

    # Pendiente representa -Ea/R
    Ea_over_R = -slope
    Ea = Ea_over_R*1.987

    return slope, intercept, Ea_over_R, r_value, Ea

def graficar_arrhenius(data_df):
    sns.set(style="white")  # Configuración del estilo de Seaborn

    # Linealizar los datos
    slope, intercept, Ea_over_R, r_value, Ea = linealizar_arrhenius(data_df['Temperature_K'].values, data_df['k'].values)

    # Graficar los datos y ajuste lineal
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data_df['1_over_T'], y=np.log(data_df['k']), label='Datos experimentales', color='blue')

    # Corregir la generación de la línea de ajuste lineal
    arrhenius_line = slope * data_df['1_over_T'].values + intercept
    plt.plot(data_df['1_over_T'].values, arrhenius_line, color='red', label='Ajuste lineal')

    plt.xlabel('1 / Temperatura (1/K)')
    plt.ylabel('ln(k)')
    plt.legend()

    plt.title(f'Ajuste Lineal de la Ecuación de Arrhenius\n$E_a/R$: {Ea_over_R:.2f}, $E_a$: {Ea:.2f}, $R^2$: {r_value**2:.2f}')

    # Imprimir valores de Ea y ordenada al origen
    print(f'Ea/R: {Ea_over_R:.4f}')
    print(f'Ea: {Ea:.4f}')
    print(f'Ordenada al origen (ln(A)): {intercept:.4f}')
    # Calcular e^x
    resultado = np.exp(intercept)

    # Imprimir el resultado
    print(f'A = {resultado}')

    plt.show()