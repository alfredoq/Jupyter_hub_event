import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import inchi, Crippen, Lipinski, Draw, rdDepictor
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import statsmodels.api as sm
import warnings
import py3Dmol
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import itertools
import itables
itables.init_notebook_mode()

warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def graficar_heatmap_pearson(df, columna_inicio, titulo='Heatmap de correlaciones de Pearson', figsize=(14,11)):
    """
    Genera un heatmap de correlaciones de Pearson absolutas desde una columna en adelante.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        columna_inicio (str): Nombre de la primera columna a incluir.
        titulo (str): Título del gráfico.
        figsize (tuple): Tamaño del gráfico.
    """
    # Obtener todas las columnas a partir de columna_inicio
    idx_inicio = df.columns.get_loc(columna_inicio)
    columnas = df.columns[idx_inicio:]

    # Calcular la matriz de correlación absoluta
    matriz_corr = df[columnas].corr(method='pearson', numeric_only=True).abs()

    # Crear y mostrar heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matriz_corr, annot=True).set(title=titulo)
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)
    plt.show()


def analizar_regresiones_lineales_simples(df, variable_objetivo, columnas_descriptores, graficar=True):
    """
    Realiza regresiones lineales simples entre una variable objetivo y una lista de descriptores.
    
    Parámetros:
        df (pd.DataFrame): Conjunto de datos.
        variable_objetivo (str): Nombre de la columna dependiente (e.g., 'IC50_value_B2_scaled').
        columnas_descriptores (list): Lista de nombres de las columnas independientes.
        graficar (bool): Si True, muestra el gráfico de regresión para cada descriptor.
    
    Retorna:
        DataFrame con Descriptor, R², Coeficiente e Intercepto ordenado por R² descendente.
    """
    resultados = []

    for var in columnas_descriptores:
        if graficar:
            plt.figure()
            sns.regplot(x=var, y=variable_objetivo, data=df).set(
                title=f'Regression plot of {var} vs {variable_objetivo}')
            plt.show()
        
        X = df[[var]]
        y = df[[variable_objetivo]]
        
        regressor = LinearRegression()
        regressor.fit(X, y)
        y_pred = regressor.predict(X)
        
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2).fit()
        
        # Extraer coeficiente y error estándar (índice 1 es la variable, 0 sería la constante)
        coef = est.params[1]
        std_err = est.bse[1]
        intercepto = est.params[0]
                
        print(est.summary())
        print('R\u00b2: %.3f' % est.rsquared)
        print('-' * 80)
        
        resultados.append([var.replace('_norm', ''), round(est.rsquared,2), coef, std_err, intercepto])

    df_resultados = pd.DataFrame(resultados, columns=['Descriptor', 'R²', 'Coeficiente', 'Std err', 'Intercepto'])
    return df_resultados.sort_values(by='R²', ascending=False)



def correlaciones_validas(subset, pearsoncorr, umbral=0.6):
    for var1, var2 in itertools.combinations(subset, 2):
        if abs(pearsoncorr.loc[var1, var2]) >= umbral:
            return False
    return True

def obtener_combinaciones_validas(df, columnas, umbral=0.6):
    """
    Devuelve una lista de combinaciones de variables que cumplen con el umbral de correlación.

    Parámetros:
        df (DataFrame): DataFrame con los datos.
        columnas (list): Lista de nombres de columnas/descriptores.
        umbral (float): Umbral de correlación para considerar combinaciones como válidas.

    Retorna:
        Lista de tuplas con combinaciones válidas.
    """
    # Calculamos la matriz de correlación una sola vez
    pearsoncorr = df[columnas].corr(method='pearson', numeric_only=True)
    pearsoncorrpost = pearsoncorr.apply(abs)

    combinaciones_validas = []
    
    for r in range(1, 7):
        for subset in itertools.combinations(columnas, r):
            if correlaciones_validas(subset, pearsoncorrpost):
                combinaciones_validas.append(subset)
    
    return combinaciones_validas

def evaluar_modelos_qsar(df, descriptores, y_col='IC50_value_nM_norm'):
    """
    Evalúa modelos de regresión lineal múltiple para combinaciones válidas de variables según un umbral de correlación.

    Parámetros:
        df (DataFrame): DataFrame con los datos.
        descriptores (list): Lista de nombres de columnas a considerar como descriptores.
        y_col (str): Nombre de la columna objetivo.

    Devuelve:
        DataFrame con resultados ordenados por R².
    """
    combis_validas = obtener_combinaciones_validas(df, descriptores)

    resultados = []

    for subset in combis_validas:
        X = df[list(subset)]
        X = sm.add_constant(X)
        y = df[y_col]

        model = sm.OLS(y, X).fit()

        resultados.append({
            "Variables": [col.replace('_norm', '') for col in subset],
            "R2": round(model.rsquared, 2),
            "Coeficientes": [round(c, 4) for c in model.params[1:]],  # sin constante
            "Errores_Estandar": [round(se, 4) for se in model.bse[1:]],
            "Intercepto": round(model.params[0], 4),
            "Cantidad_de_variables": len(subset)
        })

    resultados_df = pd.DataFrame(resultados)
    resultados_final = resultados_df.sort_values(by=["R2", "Cantidad_de_variables"], ascending=[False,True])
    return resultados_final

# Lista completa de descriptores disponibles
descList = Descriptors.descList

def calcular_descriptores_molecula(mol, seleccion=None):
    """
    Calcula descriptores de una molécula usando RDKit.

    Parámetros:
        mol (Mol): Molécula RDKit.
        seleccion (list, opcional): Lista de nombres de descriptores a calcular. Si es None, calcula todos.

    Retorna:
        dict: Diccionario con nombre y valor de cada descriptor.
    """
    resultados = {}
    for nombre, func in descList:
        if (seleccion is None) or (nombre in seleccion):
            try:
                resultados[nombre] = func(mol)
            except Exception:
                resultados[nombre] = None
    return resultados

'''def agregar_descriptores(df, smiles_col='canonical_smiles', seleccion=None, antes_de_columna=None):
    """
    Agrega descriptores RDKit al DataFrame y los ubica antes de una columna específica si se indica.

    Parámetros:
        df (DataFrame): Debe contener una columna con SMILES.
        smiles_col (str): Nombre de la columna con SMILES.
        seleccion (list, opcional): Lista de nombres de descriptores a agregar.
        antes_de_columna (str, opcional): Nombre de la columna antes de la cual insertar los descriptores.

    Retorna:
        DataFrame con columnas reordenadas.
    """
    resultados = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            resultados.append(calcular_descriptores_molecula(mol, seleccion=seleccion))
        else:
            resultados.append({nombre: None for nombre, _ in descList if (seleccion is None or nombre in seleccion)})
    
    df_descriptores = pd.DataFrame(resultados)

    df = df.reset_index(drop=True)
    df_descriptores = df_descriptores.reset_index(drop=True)
    df_final = pd.concat([df, df_descriptores], axis=1)

    # Si no se especifica columna de referencia, devolvemos tal cual
    if antes_de_columna is None or antes_de_columna not in df.columns:
        return df_final

    # Reordenamos columnas
    cols = df.columns.tolist()
    descriptores_cols = df_descriptores.columns.tolist()

    idx = cols.index(antes_de_columna)
    nuevas_orden = cols[:idx] + descriptores_cols + cols[idx:]
    
    # Evitamos duplicados (por si los descriptores están también al final)
    nuevas_orden = [col for col in nuevas_orden if col in df_final.columns]

    return df_final[nuevas_orden]'''


# Diccionario: nombre en español → nombre real del descriptor
mapa_descriptores = {
    'MolWtSinH': 'HeavyAtomMolWt',
    'CargaMax': 'MaxPartialCharge',
    'CargaMin': 'MinPartialCharge',
    'DensidadEstructural': 'FpDensityMorgan2',
    'IndiceComplejidad': 'BertzCT',
    'AreaAccesibleSolvente': 'LabuteASA',
    'RefractividadMol': 'MolMR',
    'SuperficieElectronegativa': 'PEOE_VSA10',
    'SuperficieHidrofobica': 'SlogP_VSA1',
    'AreaSuperficial': 'EState_VSA7',
    'AreaRefractiva': 'SMR_VSA6',
    'DistribucionElectronica': 'VSA_EState3'
}

def agregar_descriptores(df, smiles_col='canonical_smiles', seleccion=None, antes_de_columna=None):
    """
    Agrega descriptores RDKit al DataFrame, usando nombres en español o reales.

    Parámetros:
        df (DataFrame): Debe contener una columna con SMILES.
        smiles_col (str): Nombre de la columna con SMILES.
        seleccion (list): Lista de nombres en español definidos en `mapa_descriptores`.
        antes_de_columna (str): Columna antes de la cual insertar los descriptores.

    Retorna:
        DataFrame con columnas agregadas y renombradas.
    """
    # Traducción a nombres reales
    if seleccion:
        seleccion_real = [mapa_descriptores.get(nombre, nombre) for nombre in seleccion]
    else:
        seleccion_real = None

    resultados = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            resultados.append(calcular_descriptores_molecula(mol, seleccion=seleccion_real))
        else:
            resultados.append({desc: None for desc in seleccion_real})

    df_desc = pd.DataFrame(resultados, index=df.index)

    # Renombrar columnas si se usaron nombres amigables
    if seleccion:
        renombre = {v: k for k, v in mapa_descriptores.items() if v in df_desc.columns}
        df_desc.rename(columns=renombre, inplace=True)

    # Insertar antes de una columna específica si se indica
    if antes_de_columna and antes_de_columna in df.columns:
        idx = df.columns.get_loc(antes_de_columna)
        df_final = pd.concat([df.iloc[:, :idx], df_desc, df.iloc[:, idx:]], axis=1)
    else:
        df_final = pd.concat([df, df_desc], axis=1)

    return df_final


def normalizar_columnas_seleccionadas(df, columnas, metodo="minmax"):
    """
    Normaliza columnas numéricas especificadas.

    Parámetros:
        df (pd.DataFrame): DataFrame original.
        columnas (list): Lista de nombres de columnas a normalizar.
        metodo (str): "minmax" o "standard".

    Retorna:
        pd.DataFrame: DataFrame original + columnas normalizadas con sufijo "_norm".
    """
    datos = df[columnas]

    # Escalador elegido
    if metodo == "minmax":
        scaler = MinMaxScaler()
    elif metodo == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("El método debe ser 'minmax' o 'standard'.")

    # Normalización
    datos_escalados = scaler.fit_transform(datos)
    columnas_escaladas = [col + "_norm" for col in columnas]
    df_scaled = pd.DataFrame(datos_escalados, columns=columnas_escaladas, index=df.index)

    # Combinar con el DataFrame original
    return pd.concat([df, df_scaled], axis=1)

def mostrar_moleculas_por_id(df, ids_chembl, grupo_funcional_smiles, mols_per_row=5):
    """
    Muestra moléculas seleccionadas por ChEMBL ID, alineadas según un grupo funcional, en el orden provisto.

    Parámetros:
        df (DataFrame): Contiene columnas 'canonical_smiles', 'molecule_chembl_id', 'IC50_value_nM', 'nombre_comercial'.
        ids_chembl (list): Lista ordenada de ChEMBL IDs de las moléculas a mostrar.
        grupo_funcional_smiles (str): SMILES del grupo funcional a usar como plantilla de alineación.
        mols_per_row (int): Cantidad de moléculas por fila.
        
    """
    template = Chem.MolFromSmiles(grupo_funcional_smiles)
    rdDepictor.Compute2DCoords(template)

    mols, legends = [], []

    for chembl_id in ids_chembl:
        row = df[df['molecule_chembl_id'] == chembl_id]
        if row.empty:
            continue
        row = row.iloc[0]
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if mol:
            try:
                rdDepictor.Compute2DCoords(mol)
                rdDepictor.GenerateDepictionMatching2DStructure(mol, template)
            except:
                pass

            mols.append(mol)

            ic50 = row.get('IC50_value_nM', 'NA')
            nombre_com = row.get('nombre_comercial', 'NA')
            legend = f"{chembl_id} ({nombre_com})\nIC50: {ic50:.2f} nM" if pd.notna(nombre_com) else f"{chembl_id}\nIC50: {ic50:.2f} nM"
            legends.append(legend)

    if not mols:
        print("No se encontraron moléculas válidas para mostrar.")
        return

    
    img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=mols_per_row, subImgSize=(300, 300), useSVG=False)
    display(img)