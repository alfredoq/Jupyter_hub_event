from chembl_webresource_client.new_client import new_client
import numpy
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import inchi, Descriptors, Crippen, Lipinski, Draw, rdDepictor, AllChem
import math
import numpy as np
print(numpy.__version__)
from chemplot import Plotter
import itables
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

itables.init_notebook_mode()
warnings.filterwarnings('ignore')

def buscar_info_nombre_comercial(nombre):
    """
    Busca compuestos en ChEMBL por nombre (genérico o comercial) y devuelve
    toda la información disponible, con las columnas clave al principio.

    Parámetros:

    Retorna:
        pd.DataFrame con toda la información de ChEMBL.
    """
    resultados = new_client.molecule.filter(pref_name__icontains=nombre)
    datos = list(resultados)
    df = pd.DataFrame(datos)

    # Reordenar columnas clave al inicio
    columnas_clave = ['pref_name', 'molecule_chembl_id', 'first_approval', 'max_phase', 'molecule_structures']
    otras_columnas = [col for col in df.columns if col not in columnas_clave]
    df = df[columnas_clave + otras_columnas]
    columnas_innecesarias = ["molecule_structures","cross_references","molecule_hierarchy","molecule_synonyms"]
    df = df.drop(columns=columnas_innecesarias)

    return df


def buscar_info_target(clave):
    """
    Busca blancos terapéuticos (targets) en ChEMBL por palabra clave en el nombre.

    Parámetros:
        clave (str): Palabra clave para buscar en nombres de blancos (ej. 'adrenergic').

    Retorna:
        pd.DataFrame con información detallada sobre cada blanco que coincida.
    """
    resultados = new_client.target.filter(pref_name__icontains=clave)
    datos = list(resultados)
    df = pd.DataFrame(datos)

    # Reordenar columnas clave al inicio
    columnas_clave = ['pref_name', 'target_chembl_id', 'organism', 'target_type']
    otras_columnas = [col for col in df.columns if col not in columnas_clave]
    df = df[columnas_clave + otras_columnas]

    return df


def obtener_agonistas_antagonistas_adrenergicos(receptores, tipos_accion=["AGONIST", "ANTAGONIST"]):
    """
    Descarga y combina ligandos con actividad agonista o antagonista para múltiples receptores.

    Parámetros:
        receptores (dict): Diccionario con nombres de receptores y sus target_chembl_id.
        tipos_accion (list): Lista con tipos de acción a incluir (por defecto: agonistas y antagonistas).

    Retorna:
        DataFrame con moléculas, tipo de acción, receptor, smiles y otra info relevante.
    """
    mechanism = new_client.mechanism
    datos_completos = []

    for receptor, target_id in receptores.items():
        for tipo in tipos_accion:
            print(f"Descargando {tipo}AS para receptor {receptor}...")
            data = mechanism.filter(
                target_chembl_id=target_id,
                action_type=tipo
            ).only([
                'molecule_chembl_id', 'action_type', 'max_phase', 'target_chembl_id'
            ])
            df = pd.DataFrame(data)
            if df.empty:
                continue

            df['receptor'] = receptor  

            
            smiles = []
            nombres_comerciales = []

            for chembl_id in df['molecule_chembl_id']:
                try:
                    mol_info = new_client.molecule.filter(chembl_id=chembl_id).only(['molecule_structures', 'pref_name'])[0]
                    # SMILES
                    smiles.append(mol_info.get('molecule_structures', {}).get('canonical_smiles'))
                    # Nombre preferido
                    nombres_comerciales.append(mol_info.get('pref_name'))
                except:
                    smiles.append(None)
                    nombres_comerciales.append(None)

            df['canonical_smiles'] = smiles
            df['nombre_comercial'] = nombres_comerciales


            datos_completos.append(df)

    # Concatenar todo en un único DataFrame
    df_final = pd.concat(datos_completos, ignore_index=True)
    df_final = df_final[df_final['canonical_smiles'].notna()]  # Eliminamos entradas sin SMILES
    df_final = df_final.sort_values(by=['receptor', 'nombre_comercial'], ascending=[True, True]).reset_index(drop=True)
    df_final = df_final[df_final['max_phase'] == 4]
    
    return df_final


def dibujar_moleculas(df, receptores_seleccionados=None, tipo_accion=None,
                                    grupo_funcional_smiles=None, mols_per_image=50, mols_per_row=5,
                                    subImgSize=(350, 350)):
    """
    Dibuja estructuras 2D de moléculas con nombre comercial y tipo de acción.
    
    Parámetros:
        df (DataFrame): DataFrame con columnas ['canonical_smiles', 'molecule_chembl_id', 'action_type', 'nombre_comercial', 'target_chembl_id'].
        receptores_seleccionados (list): Lista con los receptor_chembl_id a incluir (ej: ['A1', 'B2']). Si es None, incluye todos.
        tipo_accion (str or list): 'AGONIST', 'ANTAGONIST' o lista con ambos. Si es None, incluye todos.
        grupo_funcional_smiles (str): SMILES del grupo funcional para alinear (opcional).
        mols_per_image (int): Cuántas moléculas por imagen.
        mols_per_row (int): Cuántas por fila.
        subImgSize (tuple): Tamaño de cada imagen individual.
        
    """
    # Filtrar según receptor
    if receptores_seleccionados is not None:
        df = df[df['receptor'].isin(receptores_seleccionados)]

    # Filtrar por tipo de acción
    if tipo_accion is not None:
        if isinstance(tipo_accion, str):
            tipo_accion = [tipo_accion]
        df = df[df['action_type'].isin(tipo_accion)]
    
    # Eliminar sin smiles
    df = df[df['canonical_smiles'].notna()]

    # Ordenar
    df = df.sort_values(by=['receptor', 'action_type', 'nombre_comercial'])

    # Plantilla de alineamiento si se indica
    template = None
    if grupo_funcional_smiles:
        template = Chem.MolFromSmiles(grupo_funcional_smiles)
        rdDepictor.Compute2DCoords(template)

    mols, legends = [], []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if mol:
            try:
                rdDepictor.Compute2DCoords(mol)
                if template:
                    rdDepictor.GenerateDepictionMatching2DStructure(mol, template)
            except:
                pass
            mols.append(mol)

            nombre = row.get('nombre_comercial') or ''
            accion = row.get('action_type') or ''
            receptor = row.get('receptor') or ''
            legend = f"{nombre}\n{accion} {receptor}"
            legends.append(legend)

    if not mols:
        print("No hay moléculas para mostrar.")
        return

    for i in range(math.ceil(len(mols) / mols_per_image)):
        sub_mols = mols[i*mols_per_image:(i+1)*mols_per_image]
        sub_legends = legends[i*mols_per_image:(i+1)*mols_per_image]

        img = Draw.MolsToGridImage(
            sub_mols,
            legends=sub_legends,
            molsPerRow=mols_per_row,
            subImgSize=subImgSize,
            useSVG=False
        )

        
        display(img)
        
        
def comparar_moleculas_con_template(df, ids_chembl, smiles_template, nombre_template="Template", 
                                    smiles_referencia="CNC[C@H](O)c1ccc(O)c(O)c1", 
                                    nombre_referencia="EPINEFRINA", mols_per_row=5, subImgSize=(300, 300)):
    """
    Dibuja moléculas seleccionadas junto a una molécula de referencia y un template para comparación visual, 
    alineadas con el template.

    Parámetros:
        df (DataFrame): DataFrame con columnas 'molecule_chembl_id', 'canonical_smiles', 'nombre_comercial'.
        ids_chembl (list): Lista de IDs ChEMBL de las moléculas a comparar.
        smiles_template (str): SMILES del template al que se alinearán las moléculas.
        nombre_template (str): Nombre para mostrar del template.
        smiles_referencia (str): SMILES de la molécula de referencia (ej. epinefrina).
        nombre_referencia (str): Nombre para mostrar de la molécula de referencia.
        mols_per_row (int): Número de moléculas por fila en la imagen.
        subImgSize (tuple): Tamaño de cada imagen individual.

    Retorna:
        Muestra una imagen con las moléculas seleccionadas alineadas al template.
    """
    # Crear el template y la molécula de referencia
    mol_template = Chem.MolFromSmiles(smiles_template)
    rdDepictor.Compute2DCoords(mol_template)  # Generar las coordenadas 2D para el template

    mol_ref = Chem.MolFromSmiles(smiles_referencia)
    
    
    # Alinear la molécula de referencia con el template (intentando hacer una alineación más flexible)
    try:
        rdDepictor.Compute2DCoords(mol_ref)
        rdDepictor.GenerateDepictionMatching2DStructure(mol_ref, mol_template)
    except RuntimeError:
        # Si no se puede alinear, simplemente usar la molécula de referencia tal como está
        print("No se pudo alinear la molécula de referencia con el template.")
    
    # Obtener las moléculas seleccionadas del DataFrame
    df_seleccionadas = df[df['nombre_comercial'].isin(ids_chembl)].copy()

    # Crear listas para las moléculas y sus leyendas
    moleculas = [mol_ref]
    leyendas = [nombre_referencia]

    for _, fila in df_seleccionadas.iterrows():
        mol = Chem.MolFromSmiles(fila['canonical_smiles'])
        if mol:
            # Generar las coordenadas 2D para la molécula seleccionada
            rdDepictor.Compute2DCoords(mol)
            try:
                # Alinear la molécula con el template (intentando hacerlo de forma robusta)
                rdDepictor.GenerateDepictionMatching2DStructure(mol, mol_template)
            except RuntimeError:
                # Si no se puede alinear, se mantiene la orientación original
                print(f"No se pudo alinear la molécula {fila['molecule_chembl_id']} con el template.")
                
            moleculas.append(mol)
            nombre = fila.get('nombre_comercial', '') or fila['molecule_chembl_id']
            leyendas.append(nombre)

    # Dibujar las moléculas en una imagen de cuadrícula
    imagen = Draw.MolsToGridImage(moleculas, legends=leyendas, molsPerRow=mols_per_row, subImgSize=subImgSize)
    display(imagen)

    
def calcular_propiedades(df):
    """
    Calcular propiedades de relevancia ('MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds')
    y agregarlas directamente al DataFrame.

    Parámetros:
        df (DataFrame): DataFrame que contiene la columna 'canonical_smiles' con los SMILES.
    """
    # Crear las nuevas columnas de propiedades
    propiedades = ['MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds']
    
    # Función auxiliar para calcular las propiedades de cada SMILES
    def calcular(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: 
            return [None] * 5  # Retornar valores nulos si la molécula no es válida
        return [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumRotatableBonds(mol)
        ]

    # Calcular las propiedades para cada fila y asignarlas al DataFrame
    df[propiedades] = df['canonical_smiles'].apply(calcular).apply(pd.Series)
    
    # Redondear a dos decimales
    df[propiedades] = df[propiedades].round(2)
    
    return df


def graficar_distribucion_descriptores(df, columna_inicio="MolWt", titulo="Descriptor"):
    """
    Genera gráficos individuales de violín para descriptores moleculares a partir de una columna dada hasta el final del DataFrame.
    Los gráficos se muestran en una única imagen con un máximo de tres gráficos por fila.
    También se informa la media, la mediana y los cuartiles de cada descriptor.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        columna_inicio (str): Nombre de la columna desde donde empezar a incluir variables.
        titulo (str): Título del gráfico.
    """
    # Obtener columnas desde 'columna_inicio' hasta el final
    idx_inicio = df.columns.get_loc(columna_inicio)
    variables = df.columns[idx_inicio:]
    
    # Determinar el número de filas y columnas para la figura
    num_graficos = len(variables)
    filas = (num_graficos // 3) + (1 if num_graficos % 3 != 0 else 0)
    
    # Crear la figura con subgráficos
    fig, axes = plt.subplots(filas, 3, figsize=(15, 5 * filas))
    axes = axes.flatten()  # Aplanar el arreglo para iterar fácilmente
    
    for i, var in enumerate(variables):
        ax = axes[i]
        sns.violinplot(y=df[var], ax=ax, palette="pastel")
        
        # Calcular estadísticas
        media = df[var].mean()
        mediana = df[var].median()
        cuartil_25 = df[var].quantile(0.25)
        cuartil_75 = df[var].quantile(0.75)
        
        # Mostrar estadísticas en el gráfico
        ax.axhline(media, color='r', linestyle='--', label=f'Media: {media:.2f}')
        ax.axhline(mediana, color='orange', linestyle='--', label=f'Mediana: {mediana:.2f}')
        ax.axhline(cuartil_25, color='g', linestyle='--', label=f'Cuartil 25%: {cuartil_25:.2f}')
        ax.axhline(cuartil_75, color='b', linestyle='--', label=f'Cuartil 75%: {cuartil_75:.2f}')
        
        ax.set_title(f"{titulo} - {var}")
        ax.legend()
    
    # Eliminar subgráficos vacíos (en caso de que haya menos de 3 gráficos en la última fila)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    
    
def descargar_ligandos_adrenergicos(receptores, df_ligandos):
    """
    Descarga y unifica ligandos para múltiples targets desde ChEMBL, indicando a qué subtipo de receptor
    pertenece cada compuesto, y luego agrega las columnas 'action_type' y 'nombre_comercial' desde df_ligandos
    al DataFrame de ligandos descargados.
    
    Retorna:
        DataFrame con los ligandos unificados, sus actividades y la información adicional.
    """
    # Descargar y unificar los ligandos
    todos = []

    for nombre, target_id in receptores.items():
        df = new_client.activity.filter(target_chembl_id=target_id).only([
            'type', 'units', 'relation', 'value', 'canonical_smiles', 'molecule_chembl_id', 'target_id', 'document_chembl_id'])
        df = pd.DataFrame(df)
        df['receptor'] = nombre  # Agregar columna para identificar el subtipo
        df['target_id'] = target_id  # Agregar columna para identificar el subtipo
        print(f"{len(df)} ligandos del receptor {nombre}")
        todos.append(df)

    # Unir todos los DataFrames en uno solo
    df_final = pd.concat(todos, ignore_index=True)
    df_final = df_final[df_final['canonical_smiles'].notna() & (df_final['canonical_smiles'] != '')]

    # Excluir ciertos IDs específicos
    excluir_ids = ['CHEMBL4285155', 'CHEMBL3764774', 'CHEMBL3128198']
    df_final = df_final[~df_final['molecule_chembl_id'].isin(excluir_ids)]
    df_final = df_final.reset_index(drop=True)

    # Agregar columnas adicionales desde df_ligandos
    df_ligandos = df_ligandos.drop_duplicates(subset='molecule_chembl_id', keep='first')
    df_final = pd.merge(df_final, df_ligandos[['molecule_chembl_id', 'action_type', 'nombre_comercial']], 
                        on='molecule_chembl_id', how='left')

    # Asignar manualmente los valores específicos para CHEMBL434 y CHEMBL679
    df_final.loc[df_final['molecule_chembl_id'] == 'CHEMBL434', 'action_type'] = 'AGONIST'
    df_final.loc[df_final['molecule_chembl_id'] == 'CHEMBL434', 'nombre_comercial'] = 'ISOPROTERENOL'
    df_final.loc[df_final['molecule_chembl_id'] == 'CHEMBL679', 'action_type'] = 'AGONIST'
    df_final.loc[df_final['molecule_chembl_id'] == 'CHEMBL679', 'nombre_comercial'] = 'EPINEPHRINE'

    # Eliminar filas sin coincidencia
    df_final = df_final.reset_index(drop=True)
    
    columnas_ordenadas = [
    'receptor', 'molecule_chembl_id', 'canonical_smiles', 'type',
    'relation', 'value', 'units', 'target_id', 'action_type', 'nombre_comercial','document_chembl_id'
    ]

    # Reordenar columnas si todas existen
    df_final = df_final[[col for col in columnas_ordenadas if col in df_final.columns]]
    
    return df_final


def contar_ligandos_unicos_por_receptor(df):
    """
    Cuenta la cantidad de ligandos únicos (molecule_chembl_id) por cada receptor.

    Parámetros:
        df (DataFrame): DataFrame con columnas 'receptor' y 'molecule_chembl_id'.

    Retorna:
        DataFrame con el conteo por receptor.
    """    
    print(df.groupby('receptor')['molecule_chembl_id'].nunique().reset_index(name='ligandos_unicos'))
    return 


def actividad_por_receptor(df, tipo_actividad="IC50"):
    """
    Retorna un DataFrame con una fila por molécula y columnas para cada receptor, mostrando
    el valor de actividad indicado (por ejemplo, IC50) en cada receptor donde esté disponible.

    Filtra valores no numéricos y promedia si hay múltiples reportes.
    """
    # Filtrar por tipo de actividad deseado
    
    df_filtrado = df[df['type'] == tipo_actividad].copy()

    # Convertir la columna 'value' a numérico (invalid entries serán NaN)
    df_filtrado['value'] = pd.to_numeric(df_filtrado['value'], errors='coerce')

    # Quitar filas con valores inválidos
    df_filtrado = df_filtrado.dropna(subset=['value'])

    # Agrupar para promediar múltiples valores por receptor
    df_prom = df_filtrado.groupby(['molecule_chembl_id', 'receptor'])['value'].mean().reset_index()

    # Reorganizar como tabla pivote
    df_pivot = df_prom.pivot(index='molecule_chembl_id', columns='receptor', values='value')

    # Filtrar ligandos con actividad en más de un receptor
    df_pivot = df_pivot[df_pivot.notna().sum(axis=1) > 1]

    return df_pivot


def determinar_chemspace(df, smiles_col='canonical_smiles', random_state=4):
    """
    Calcula las coordenadas UMAP para un conjunto de moléculas y las agrega al DataFrame original.

    Parámetros:
        df (pd.DataFrame): DataFrame original que contiene SMILES.
        smiles_col (str): Nombre de la columna que contiene los SMILES.
        random_state (int): Seed para UMAP reproducible.

    Retorna:
        pd.DataFrame: DataFrame con columnas 'UMAP-1' y 'UMAP-2' agregadas.
    """
    cp = Plotter.from_smiles(df[smiles_col], target=df["receptor"], target_type="C", sim_type="structural")
    chemical_space_data = cp.umap(random_state=random_state).drop(columns=['target'])

    df_umap = df.join(chemical_space_data)
    return df_umap



def graficar_espacio_quimico(df, x='UMAP-1', y='UMAP-2', receptor_col='receptor',
                              titulo='Espacio químico de ligandos adrenérgicos',
                              figsize=(20, 12), alpha=0.9):
    """
    Grafica el espacio químico UMAP con seaborn, coloreando por subtipo de receptor.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene coordenadas UMAP y columna de receptor.
        x, y (str): Nombres de las columnas UMAP.
        receptor_col (str): Columna con la categoría de receptor.
        orden_receptores (list): Orden de los receptores en la leyenda y visualización.
        titulo (str): Título del gráfico.
        figsize (tuple): Tamaño de la figura.
        alpha (float): Transparencia de los puntos.
    """
        
    # Invertir para que los últimos se dibujen primero (queden "debajo")
    df = df.sort_values(by=receptor_col, ascending=False)

    # Graficar
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x, y=y, hue=receptor_col, alpha=alpha)
    plt.title(titulo)
    plt.show()
    
    
def separar_alfa_beta(df, receptor_col='receptor'):
    """
    Separa los compuestos del DataFrame en ligandos para receptores alfa (A1, A2) y beta (B1, B2).

    Parámetros:
        df (pd.DataFrame): DataFrame con columna que indica el subtipo de receptor.
        receptor_col (str): Nombre de la columna que contiene el subtipo (por defecto: 'receptor').

    Retorna:
        df_alfa, df_beta: Dos DataFrames, uno con receptores alfa y otro con beta.
    """
    alfa = ['A1', 'A2']
    beta = ['B1', 'B2']

    df_alfa = df[df[receptor_col].isin(alfa)].copy()
    df_beta = df[df[receptor_col].isin(beta)].copy()

    return df_alfa.reset_index(drop=True), df_beta.reset_index(drop=True)


def graficar_espacio_quimico_con_filtro(df_completo, df_coloreado, x='UMAP-1', y='UMAP-2',
                                         receptor_col='receptor',
                                         titulo='Espacio químico de ligandos adrenérgicos',
                                         figsize=(20, 12), alpha=0.9):
    """
    Grafica todos los puntos del espacio químico UMAP.
    Los que están en df_coloreado se pintan por receptor, y el resto en gris.
    
    Parámetros:
        df_completo (pd.DataFrame): DataFrame con todos los compuestos.
        df_coloreado (pd.DataFrame): Subconjunto que se colorea por receptor.
        x, y (str): Columnas UMAP.
        receptor_col (str): Columna con el tipo de receptor.
        titulo (str): Título del gráfico.
        figsize (tuple): Tamaño de figura.
        alpha (float): Transparencia de puntos.
    """
    # Asegurarse de que ambas tablas tengan la misma columna clave
    key = 'molecule_chembl_id'
    
    # Separar los que no están en el df_coloreado
    ids_coloreados = set(df_coloreado[key])
    df_gris = df_completo[~df_completo[key].isin(ids_coloreados)].copy()
    df_color = df_completo[df_completo[key].isin(ids_coloreados)].copy()

    # Agregar columna para graficar en gris
    df_gris[receptor_col] = 'Otros'

    # Comenzar gráfico
    plt.figure(figsize=figsize)

    # Dibujar primero en gris
    sns.scatterplot(data=df_gris, x=x, y=y, color='gray', alpha=alpha)

    # Luego los coloreados, por tipo de receptor
    sns.scatterplot(data=df_color, x=x, y=y, color='red', alpha=alpha)

    plt.title(titulo)
    #plt.legend(title='Receptor', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    
def separar_beta1_beta2(df, receptor_col='receptor'):
    """
    Separa los compuestos del DataFrame en ligandos para receptores β1 y β2.

    Parámetros:
        df (pd.DataFrame): DataFrame con columna que indica el subtipo de receptor.
        receptor_col (str): Nombre de la columna que contiene el subtipo (por defecto: 'receptor').

    Retorna:
        df_beta1, df_beta2: Dos DataFrames, uno con receptores β1 y otro con β2.
    """
    df_beta1 = df[df[receptor_col] == 'B1'].copy()
    df_beta2 = df[df[receptor_col] == 'B2'].copy()

    return df_beta1.reset_index(drop=True), df_beta2.reset_index(drop=True)


def aislar_actividad_biologica(df, tipo='IC50'):
    """
    Convierte unidades a nM, filtra por tipo de actividad y promedia valores por ligando,
    conservando las columnas adicionales del DataFrame original.

    Parámetros:
        df (pd.DataFrame): base con columnas ['type', 'value', 'units', 'molecule_chembl_id', 'canonical_smiles', ...]
        tipo (str): tipo de actividad a filtrar (ej. 'IC50')

    Retorna:
        pd.DataFrame con valores promedio y columnas originales.
    """
    
    conversion_dict = {
    'nM': 1, "10'-9M": 1, 'nmol/L': 1, 'nmol/l': 1, 'nM l-1 hr-1': 1,
    'uM': 1000, "10'3nM": 1000, "10'-6 mol/L": 1000,
    'M': 1_000_000, "10'-8M": 10, "10'-7M": 100, "10'4nM": 10_000,
    "10'-11M": 0.01, "10'-10M": 0.1, "10'-4microM": 0.1, 'pM': 0.001
    }
    
    if df.empty:
        return pd.DataFrame()

    # Filtrar por tipo de actividad
    df_tipo = df[df['type'] == tipo].copy()

    # Asegurar que 'value' sea numérico
    df_tipo['value'] = pd.to_numeric(df_tipo['value'], errors='coerce')

    # Eliminar filas sin valor o unidad
    df_tipo = df_tipo.dropna(subset=['value', 'units'])

    # Aplicar conversión de unidades
    def convertir(row):
        unidad = row['units']
        if unidad in conversion_dict:
            return row['value'] * conversion_dict[unidad]
        return None

    df_tipo[tipo + '_value_nM'] = df_tipo.apply(convertir, axis=1)
    df_tipo = df_tipo.dropna(subset=[tipo + '_value_nM'])

    # Calcular promedio por molécula
    df_prom = df_tipo.groupby('molecule_chembl_id')[tipo + '_value_nM'].mean().reset_index()

    # Tomar la primera fila de cada molécula para conservar info original
    df_resto = df_tipo.sort_values(by='value').drop_duplicates('molecule_chembl_id')

    # Merge para reconstruir dataframe completo
    df_final = pd.merge(df_resto.drop(columns=[tipo + '_value_nM', 'type', 'units', 'value']), df_prom, on='molecule_chembl_id', how='left')
    
    orden_deseado = [
    'receptor', 'molecule_chembl_id', 'canonical_smiles', 'target_id',
    'action_type', 'nombre_comercial', 'document_chembl_id', 'UMAP-1', 'UMAP-2',
    'relation', tipo + '_value_nM']

    df_final = df_final[[col for col in orden_deseado if col in df_final.columns]]


    return df_final


def filtrar_por_grupo_funcional(df, smarts, incluir=True):
    """
    Filtrar moléculas en base presencia o ausencia de un determinado grupo funcional.

    Parámetros:
        df (str): base de datos conteniendo moléculas de interés.
        smarts (str): grupo funcional expresado en nomenclatura SMARTS a ser filtrado.
        incluir (bool): True si queremos que CONTENGA el grupo, False si queremos quedarnos con aquellas que NO lo contengan.
    """
    patron = Chem.MolFromSmarts(smarts)
    mask = df['canonical_smiles'].apply(lambda s: Chem.MolFromSmiles(s).HasSubstructMatch(patron) if Chem.MolFromSmiles(s) else False)
    df_filtrado = df[mask] if incluir else df[~mask]
    print(f"✅ {len(df_filtrado)} moléculas con {smarts}.")
    return df_filtrado


def ver_rangos_propiedades(df):
    """
    Obtener información de los valores máximos y mínimos de todas las propiedades de relevancia ('MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds') calculadas.

    Parámetros:
        df (str): base de datos conteniendo datos de las propiedades.
    """
    propiedades = ['MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds']
    for prop in propiedades:
        if prop in df.columns:
            print(f"🔹 {prop}: {df[prop].min()} – {df[prop].max()}")
            

def filtrar_multiples_propiedades(df, criterios):
    """
    Filtrar moléculas en base a MÚLTIPLES propiedades de relevancia ('MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds').

    Parámetros:
        df (str): base de datos conteniendo datos de las propiedades.
        criterios (diccionario): diccionario conteniendo los criterios de filtración.
        
    """
    df_filtrado = df.copy()
    for prop, (min_val, max_val) in criterios.items():
        if prop not in df.columns: continue
        if min_val is not None:
            df_filtrado = df_filtrado[df_filtrado[prop] >= min_val]
        if max_val is not None:
            df_filtrado = df_filtrado[df_filtrado[prop] <= max_val]
    return df_filtrado

def graficar_distribucion_descriptores_all(df, columna_inicio, titulo="Distribución de descriptores moleculares"):
    """
    Genera un gráfico de violín para descriptores moleculares a partir de una columna dada hasta el final del DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        columna_inicio (str): Nombre de la columna desde donde empezar a incluir variables.
        titulo (str): Título del gráfico.
    """
    if "norm" in columna_inicio:
        df = df[df["IC50_value_nM"]<3100]
        df = df.drop(columns=[col for col in df.columns if col.endswith('_norm')])
        df = normalizar_desde_columna(df, "IC50_value_nM")
    # Obtener columnas desde 'columna_inicio' hasta el final
    idx_inicio = df.columns.get_loc(columna_inicio)
    variables = df.columns[idx_inicio:]
    
    df_long = df[variables].melt(var_name="Descriptor", value_name="Valor")
    
    plt.figure(figsize=(20, 8))
    sns.violinplot(x="Descriptor", y="Valor", data=df_long)
    plt.xticks(rotation=45)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def normalizar_desde_columna(df, columna_inicio, metodo="minmax"):
    """
    Normaliza todas las columnas numéricas a partir de una columna específica.
    
    Parámetros:
        df (pd.DataFrame): DataFrame original.
        columna_inicio (str): Nombre de la columna desde donde empezar a normalizar.
        metodo (str): "minmax" o "standard" para elegir el tipo de escalado.

    Retorna:
        pd.DataFrame: DataFrame original + columnas normalizadas con sufijo "_scaled".
    """
    idx_inicio = df.columns.get_loc(columna_inicio)
    columnas_a_normalizar = df.columns[idx_inicio:]
    datos = df[columnas_a_normalizar]

    # Escalador elegido
    if metodo == "minmax":
        scaler = MinMaxScaler()
    elif metodo == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    else:
        raise ValueError("El método debe ser 'minmax' o 'standard'.")

    # Normalización
    datos_escalados = scaler.fit_transform(datos)
    columnas_escaladas = [col + "_norm" for col in columnas_a_normalizar]
    df_scaled = pd.DataFrame(datos_escalados, columns=columnas_escaladas, index=df.index)
    
    # Combinar con el DataFrame original
    return pd.concat([df, df_scaled], axis=1)
    


def mostrar_y_guardar_moleculas_alineadas(df, grupo_funcional_smiles, mols_per_image=50, mols_per_row=10, output_folder='imagenes'):
    """
    Muestra y guarda imágenes de moléculas alineadas según un grupo funcional, ordenadas por IC50.

    Parámetros:
        df (DataFrame): Contiene al menos las columnas 'canonical_smiles', 'molecule_chembl_id', 'ExactMolWt', 'IC50_value_nM'.
        grupo_funcional_smiles (str): SMILES del grupo funcional a usar como plantilla de alineación.
        mols_per_image (int): Cantidad de moléculas por imagen.
        mols_per_row (int): Cantidad de moléculas por fila.
        output_folder (str): Carpeta donde se guardarán las imágenes generadas.
    """
    # Crear carpeta si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Ordenar por peso molecular
    df = df.sort_values('IC50_value_nM')

    # Crear molécula plantilla
    template = Chem.MolFromSmiles(grupo_funcional_smiles)
    rdDepictor.Compute2DCoords(template)

    mols, legends = [], []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if mol:
            try:
                rdDepictor.Compute2DCoords(mol)
                rdDepictor.GenerateDepictionMatching2DStructure(mol, template)
            except:
                pass  # Si falla, usa coordenadas estándar

            mols.append(mol)

            chembl_id = row.get('molecule_chembl_id', 'NA')
            ic50 = row.get('IC50_value_nM', 'NA')
            nombre_com = row.get('nombre_comercial', 'NA')
            legend = f"{chembl_id} ({nombre_com}) \nIC50: {ic50:.2f} nM" if pd.notna(nombre_com) else f"{chembl_id}\nIC50: {ic50:.2f} nM"
            legends.append(legend)

    # Mostrar y guardar por bloques
    total = len(mols)
    num_images = math.ceil(total / mols_per_image)

    for i in range(num_images):
        start = i * mols_per_image
        end = min((i + 1) * mols_per_image, total)

        img = Draw.MolsToGridImage(
            mols[start:end],
            molsPerRow=mols_per_row,
            legends=legends[start:end],
            subImgSize=(250, 250),
            returnPNG=False
        )

        # Mostrar en el notebook
        plt.figure(figsize=(mols_per_row * 3, math.ceil((end - start) / mols_per_row) * 3))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Guardar imagen
        output_path = os.path.join(output_folder, f"mol_imagen_{i+1}.png")
        img.save(output_path)
        
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
    

def combinar_dataframes(df_base, archivo_csv):
    """
    Combina df_base con un DataFrame cargado desde un archivo CSV,
    asegurando que todas las columnas de df_base estén presentes.
    
    Parámetros:
        df_base (pd.DataFrame): DataFrame original con estructura completa.
        archivo_csv (str): Ruta al archivo CSV con el segundo DataFrame.

    Retorna:
        pd.DataFrame combinado.
    """
    # Cargar CSV
    df_nuevo = pd.read_csv(archivo_csv)
    
    # Identificar columnas faltantes en cada DataFrame
    columnas_base = set(df_base.columns)
    columnas_nuevo = set(df_nuevo.columns)

    faltantes_en_nuevo = columnas_base - columnas_nuevo
    faltantes_en_base = columnas_nuevo - columnas_base

    # Agregar columnas faltantes con NaNs
    for col in faltantes_en_nuevo:
        df_nuevo[col] = pd.NA
    for col in faltantes_en_base:
        df_base[col] = pd.NA

    # Reordenar columnas para que coincidan
    df_nuevo = df_nuevo[df_base.columns]

    # Combinar
    df_combinado = pd.concat([df_base, df_nuevo], ignore_index=True)
    return df_combinado