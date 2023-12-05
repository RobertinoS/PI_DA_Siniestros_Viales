## FUNCIONES DE UTILIDAD PARA EL ETL Y EDA
# Importaciones
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from textblob import TextBlob
import re


def verificar_duplicados(df):
    # Identificar duplicados por filas
    duplicates = df[df.duplicated(keep=False)]
    duplicates = duplicates.sort_values(by=list(df.columns))
    
    # Imprimir la cantidad de filas duplicadas
    print(f"La cantidad de filas duplicadas es: {len(duplicates) // 2}")
    
    print("Ubicación de los datos duplicados:")
    prev_row = None
    for index, row in duplicates.iterrows():
        if row.equals(prev_row):
            print(f"La fila {prev_index} es idéntica a la fila {index}")
        prev_row = row
        prev_index = index


def verifica_tipo_y_nulos(df):
    '''
    Realiza un análisis de los tipos de datos y la presencia de valores nulos en un DataFrame.

    Esta función toma un DataFrame como entrada y devuelve un resumen que incluye información sobre
    los tipos de datos en cada columna, el porcentaje de valores no nulos y nulos, así como la
    cantidad de valores nulos por columna.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        pandas.DataFrame: Un DataFrame que contiene el resumen de cada columna, incluyendo:
        - 'nombre_campo': Nombre de cada columna.
        - 'tipo_datos': Tipos de datos únicos presentes en cada columna.
        - 'no_nulos_%': Porcentaje de valores no nulos en cada columna.
        - 'nulos_%': Porcentaje de valores nulos en cada columna.
        - 'nulos': Cantidad de valores nulos en cada columna.
    '''

    mi_dict = {"nombre_campo": [], "tipo_datos": [], "no_nulos_%": [], "nulos_%": [], "nulos": []}

    for columna in df.columns:
        porcentaje_no_nulos = (df[columna].count() / len(df)) * 100
        mi_dict["nombre_campo"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
        mi_dict["no_nulos_%"].append(round(porcentaje_no_nulos, 2))
        mi_dict["nulos_%"].append(round(100-porcentaje_no_nulos, 2))
        mi_dict["nulos"].append(df[columna].isnull().sum())

    df_info = pd.DataFrame(mi_dict)
        
    return df_info

def verificar_datos_unicos(df): #determina si estan repetidos 
    columnas_df = df.columns.tolist()
    tipos_datos = [float, int, str]
    for columna in columnas_df:
        for tipo in tipos_datos:
            filtro = df[columna][df[columna].apply(lambda x: isinstance(x, tipo))]
            valores_unicos = filtro.unique()
            print(columna, " (", tipo.__name__, ") ", len(valores_unicos),": ", valores_unicos)
        print("")



def verifica_duplicados_por_columna(df, columna): 
    '''
    Verifica y muestra filas duplicadas en un DataFrame basado en una columna específica.

    Esta función toma como entrada un DataFrame y el nombre de una columna específica.
    Luego, identifica las filas duplicadas basadas en el contenido de la columna especificada,
    las filtra y las ordena para una comparación más sencilla.

    Parameters:
        df (pandas.DataFrame): El DataFrame en el que se buscarán filas duplicadas.
        columna (str): El nombre de la columna basada en la cual se verificarán las duplicaciones.

    Returns:
        pandas.DataFrame or str: Un DataFrame que contiene las filas duplicadas filtradas y ordenadas,
        listas para su inspección y comparación, o el mensaje "No hay duplicados" si no se encuentran duplicados.
    '''
    # Se filtran las filas duplicadas
    duplicated_rows = df[df.duplicated(subset=columna, keep=False)]
    if duplicated_rows.empty:
        return "No hay duplicados"
    
    # se ordenan las filas duplicadas para comparar entre sí
    duplicated_rows_sorted = duplicated_rows.sort_values(by=columna)
    return duplicated_rows_sorted

def analisis_sentimiento(review):
    '''
    Realiza un análisis de sentimiento en un texto dado y devuelve un valor numérico que representa el sentimiento.

    Esta función utiliza la librería TextBlob para analizar el sentimiento en un texto dado y
    asigna un valor numérico de acuerdo a la polaridad del sentimiento.

    Parameters:
        review (str): El texto que se va a analizar para determinar su sentimiento.

    Returns:
        int: Un valor numérico que representa el sentimiento del texto:
             - 0 para sentimiento negativo.
             - 1 para sentimiento neutral o no clasificable.
             - 2 para sentimiento positivo.
    '''
    if review is None:
        return 1
    analysis = TextBlob(review)
    polarity = analysis.sentiment.polarity
    if polarity < -0.2:
        return 0  
    elif polarity > 0.2: 
        return 2 
    else:
        return 1 
    
def ejemplos_review_por_sentimiento(reviews, sentiments):
    '''
    Imprime ejemplos de reviews para cada categoría de análisis de sentimiento.

    Esta función recibe dos listas paralelas, `reviews` que contiene los textos de las reviews
    y `sentiments` que contiene los valores de sentimiento correspondientes a cada review.
    
    Parameters:
        reviews (list): Una lista de strings que representan los textos de las reviews.
        sentiments (list): Una lista de enteros que representan los valores de sentimiento
                          asociados a cada review (0, 1, o 2).

    Returns:
        None: La función imprime los ejemplos de reviews para cada categoría de sentimiento.
    '''
    for sentiment_value in range(3):
        print(f"Para la categoría de análisis de sentimiento {sentiment_value} se tienen estos ejemplos de reviews:")
        sentiment_reviews = [review for review, sentiment in zip(reviews, sentiments) if sentiment == sentiment_value]
        
        for i, review in enumerate(sentiment_reviews[:3], start=1):
            print(f"Review {i}: {review}")
        
        print("\n")
    

def extrae_anio(fecha):    
    '''
    Extrae el año de una fecha en formato 'yyyy-mm-dd' y maneja valores nulos.

    Esta función toma como entrada una fecha en formato 'yyyy-mm-dd' y devuelve el año de la fecha si
    el dato es válido. Si la fecha es nula o inconsistente, devuelve 'Dato no disponible'.

    Parameters:
        fecha (str or float or None): La fecha en formato 'yyyy-mm-dd'.

    Returns:
        str: El año de la fecha si es válido, 'Dato no disponible' si es nula o el formato es incorrecto.
    '''
    if pd.notna(fecha):
        if re.match(r'^\d{4}-\d{2}-\d{2}$', fecha):
            return fecha.split('-')[0]
    return 'Dato no disponible'

def columna_rating(fila):               
    ''' 
    
1: si sa=0 y r=True o False
2: si sa=1 y r=False
3: si sa=1 y r=True
4: si sa=2 y r=False
5: si sa=2 y r=True
'''
    if fila["sentiment_analysis"] == 0 and not fila["reviews_recommend"]:
        return 1
    elif fila["sentiment_analysis"] == 1 and fila["reviews_recommend"]:
        return 2
    elif fila["sentiment_analysis"] == 2 and fila["reviews_recommend"]:
        return 2
    else:
        return None


def convertir_fecha(cadena_fecha):      
    '''
    Convierte una cadena de fecha en un formato específico a otro formato de fecha.
    
    Args:
    cadena_fecha (str): Cadena de fecha en el formato "Month Day, Year" (por ejemplo, "September 1, 2023").
    
    Returns:
    str: Cadena de fecha en el formato "YYYY-MM-DD" o un mensaje de error si la cadena no cumple el formato esperado.
    '''
    match = re.search(r'(\w+\s\d{1,2},\s\d{4})', cadena_fecha)
    if match:
        fecha_str = match.group(1)
        try:
            fecha_dt = pd.to_datetime(fecha_str)
            return fecha_dt.strftime('%Y-%m-%d')
        except:
            return 'Fecha inválida'
    else:
        return 'Formato inválido'

def convertir_a_time(x):
    '''
    Convierte un valor a un objeto de tiempo (time) de Python si es posible.

    Esta función acepta diferentes tipos de entrada y trata de convertirlos en objetos de tiempo (time) de Python.
    Si la conversión no es posible, devuelve None.

    Parameters:
        x (str, datetime, or any): El valor que se desea convertir a un objeto de tiempo (time).

    Returns:
        datetime.time or None: Un objeto de tiempo (time) de Python si la conversión es exitosa,
        o None si no es posible realizar la conversión.
    '''
    if isinstance(x, str):
        try:
            return datetime.strptime(x, "%H:%M:%S").time()
        except ValueError:
            return None
    elif isinstance(x, datetime):
        return x.time()
    return x

def convierte_a_flotante(value):    
    '''
    Reemplaza valores no numéricos y nulos en una columna con 0.0.

    Esta función toma un valor como entrada y trata de convertirlo a un número float.
    Si la conversión es exitosa, el valor numérico se mantiene. Si la conversión falla o
    el valor es nulo, se devuelve 0.0 en su lugar.

    Parameters:
        value: El valor que se va a intentar convertir a un número float o nulo.

    Returns:
        float: El valor numérico si la conversión es exitosa o nulo, o 0.0 si la conversión falla.
    '''
    if pd.isna(value):
        return 0.0
    try:
        float_value = float(value)
        return float_value
    except:
        return 0.0
    
def resumen_cant_porcentaje(df, columna):           
    '''
    Cuanta la cantidad de True/False luego calcula el porcentaje.

    Parameters:
    - df (DataFrame): El DataFrame que contiene los datos.
    - columna (str): El nombre de la columna en el DataFrame para la cual se desea generar el resumen.

    Returns:
    DataFrame: Un DataFrame que resume la cantidad y el porcentaje de True/False en la columna especificada.
    '''
    # Cuanta la cantidad de True/False luego calcula el porcentaje
    counts = df[columna].value_counts()
    percentages = round(100 * counts / len(df),2)
    # Crea un dataframe con el resumen
    df_results = pd.DataFrame({
        "Cantidad": counts,
        "Porcentaje": percentages
    })
    return df_results

def bigote_max(columna):
    '''
    Calcula el valor del bigote superior y la cantidad de valores atípicos en una columna.

    Parameters:
    - columna (pandas.Series): La columna de datos para la cual se desea calcular el bigote superior y encontrar valores atípicos.

    Returns:
    None
    '''
    # Cuartiles
    q1 = columna.describe()[4]
    q3 = columna.describe()[6]

    # Valor del vigote
    bigote_max = round(q3 + 1.5*(q3 - q1), 2)
    print(f'El bigote superior de la variable {columna.name} se ubica en:', bigote_max)

    # Cantidad de atípicos
    print(f'Hay {(columna > bigote_max).sum()} valores atípicos en la variable {columna.name}')
    
def valor_frecuente(df, columna):
    '''
    Imputa los valores faltantes en una columna de un DataFrame con el valor más frecuente.

    Esta función reemplaza los valores "SD" con NaN en la columna especificada,
    luego calcula el valor más frecuente en esa columna y utiliza ese valor
    para imputar los valores faltantes (NaN).

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene la columna a ser imputada.
        columna (str): El nombre de la columna en la que se realizará la imputación.

    Returns:
        None
    '''
    # Se reemplaza "SD" con NaN en la columna
    df[columna] = df[columna].replace('SD', pd.NA)

    # Se calcula el valor más frecuente en la columna
    val_frecuente = df[columna].mode().iloc[0]
    print(f'El valor mas frecuente es: {val_frecuente}')

    # Se imputan los valores NaN con el valor más frecuente
    df[columna].fillna(val_frecuente, inplace=True)
    
def edad_media_segun_sexo(df):
    '''
    Imputa valores faltantes en la columna 'Edad' utilizando la edad promedio según el género.

    Esta función reemplaza los valores "SD" con NaN en la columna 'Edad', calcula la edad promedio
    para cada grupo de género (Femenino y Masculino), imprime los promedios calculados y
    luego llena los valores faltantes en la columna 'Edad' utilizando el promedio correspondiente
    al género al que pertenece cada fila en el DataFrame.

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene la columna 'Edad' a ser imputada.

    Returns:
        None
    '''
    
    # Se reemplaza "SD" con NaN en la columna 'edad'
    df['Edad'] = df['Edad'].replace('SD', pd.NA)

    # Se calcula el promedio de edad para cada grupo de género
    prom_genero = df.groupby('Sexo')['Edad'].mean()
    print(f'La edad promedio de Femenino es {round(prom_genero["FEMENINO"])} y de Masculino es {round(prom_genero["MASCULINO"])}')

    # Se llenan los valores NaN en la columna 'edad' utilizando el promedio correspondiente al género
    df['Edad'] = df.apply(lambda row: prom_genero[row['Sexo']] if pd.isna(row['Edad']) else row['Edad'], axis=1)
    # Lo convierte a entero
    df['Edad'] = df['Edad'].astype(int)
    
def intervalo_dia(hora):
  """
  Devuelve la categoría de tiempo correspondiente a la hora proporcionada.

  Parameters:
    hora: La hora a clasificar.

  Returns:
    La categoría de tiempo correspondiente.
  """
  if hora.hour >= 6 and hora.hour <= 10:
    return "Mañana"
  elif hora.hour >= 11 and hora.hour <= 13:
    return "Medio día"
  elif hora.hour >= 14 and hora.hour <= 18:
    return "Tarde"
  elif hora.hour >= 19 and hora.hour <= 23:
    return "Noche"
  else:
    return "Madrugada"
    
def accidentes_por_intervalo(horarios):
    '''
    Calcula la cantidad de accidentes por intervalo de tiempo y muestra un gráfico de barras.

    Esta función toma un DataFrame que contiene una columna 'Hora' y utiliza la función
    'intervalo_dia' para crear la columna 'Intervalo'. Luego, cuenta
    la cantidad de accidentes por cada intervalo de tiempo, calcula los porcentajes y
    genera un gráfico de barras que muestra la distribución de accidentes por intervalo de tiempo.

    Parameters:
        horarios (pandas.DataFrame): El DataFrame que contiene la información de los accidentes.

    Returns:
        None
    '''
    # Se aplica la función intervalo_dia para crear la columna 'Intervalo'
    horarios['Intervalo'] = horarios['Hora'].apply(intervalo_dia)

    # Se cuenta la cantidad de accidentes por intervalo de tiempo
    datos = horarios['Intervalo'].value_counts().reset_index()
    datos.columns = ['Intervalo', 'Num Accidentes']

    # Se calculan los porcentajes
    total_accidentes = datos['Num Accidentes'].sum()
    datos['Porcentaje'] = (datos['Num Accidentes'] / total_accidentes) * 100
    
    # Se crea el gráfico de barras
    plt.figure(figsize=(6, 4))
    ax = plt.barh(datos['Intervalo'], datos['Num Accidentes'])

    plt.title('Número de Accidentes por Intervalo de Tiempo') 
    plt.ylabel('Intervalo de Tiempo') 
    plt.xlabel('Número de Accidentes')

    # Se agrega las cantidades en las barras
    for i in range(len(datos)):
        plt.text(datos['Num Accidentes'][i], i, datos['Num Accidentes'][i], va='center')

    # Se muestra el gráfico
    plt.show()

def accidentes_por_hora(datos):
    '''
    Genera un gráfico de barras que muestra la cantidad de accidentes por hora del día.

    Parameters:
        datos: El conjunto de datos de accidentes.

    Returns:
        Un gráfico de barras.
    '''
    # Se extrae la hora del día de la columna 'hora'
    datos['HoraDia'] = datos['Hora'].apply(lambda x: x.hour)

    # Se cuenta la cantidad de accidentes por hora del día
    datos_accidentes = datos['HoraDia'].value_counts().reset_index()
    datos_accidentes.columns = ['HoraDia', 'NumAccidentes']

    # Se ordena los datos por hora del día
    datos_accidentes = datos_accidentes.sort_values(by='HoraDia')

    # Se crea el gráfico de barras
    plt.figure(figsize=(12, 4))
    ax = sns.barplot(x='HoraDia', y='NumAccidentes', data=datos_accidentes)

    plt.title('Número de Accidentes por Hora del Día') 
    plt.xlabel('Hora del Día') 
    plt.ylabel('Número de Accidentes')

    # Se agrega las cantidades en las barras
    for i, v in enumerate(datos_accidentes['NumAccidentes']):
        ax.text(i, v + 0.5, str(v), color='black', ha='center')

    # Se muestra el gráfico
    plt.show()
    
def victimas_por_dia(datos):
    '''
    Crea un gráfico de barras que muestra la cantidad de víctimas de accidentes por día de la semana.

    Esta función toma un DataFrame que contiene datos de accidentes, convierte la columna 'Fecha' a tipo de dato
    datetime si aún no lo es, extrae el día de la semana (0 = lunes, 6 = domingo), mapea el número del día
    de la semana a su nombre, cuenta la cantidad de accidentes por día de la semana y crea un gráfico de barras
    que muestra la cantidad de víctimas para cada día de la semana.

    Parameters:
        datos (pandas.DataFrame): El DataFrame que contiene los datos de accidentes con una columna 'Fecha'.

    Returns:
        None
    '''
    # Se convierte la columna 'fecha' a tipo de dato datetime
    datos['Fecha'] = pd.to_datetime(datos['Fecha'])
    
    # Se extrae el día de la semana (0 = lunes, 6 = domingo)
    datos['DiaSemanaNum'] = datos['Fecha'].dt.dayofweek
    
    # Se mapea el número del día de la semana a su nombre
    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    datos['DiaSemana'] = datos['DiaSemanaNum'].map(lambda x: dias_semana[x])
    
    # Se cuenta la cantidad de accidentes por día de la semana
    data = datos.groupby('DiaSemana').agg({'Cantidad de victimas':'sum'}).reset_index()
      
    # Se crea el gráfico de barras
    plt.figure(figsize=(6, 3))
    plt.barh(data['DiaSemana'], data['Cantidad de victimas'])

    plt.title('Cantidad de Accidentes por Día de la Semana') 
    plt.ylabel('Día de la Semana') 
    plt.xlabel('Cantidad de Accidentes')
    
    # Se muestran datos resumen
    print(f'El día de la semana con menor cantidad de víctimas tiene {data.min()[1]} víctimas')
    print(f'El día de la semana con mayor cantidad de víctimas tiene {data.max()[1]} víctimas')
    print(f'La diferencia porcentual es de {round((data.max()[1] - data.min()[1]) / data.min()[1] * 100,2)}')
    
    # Se muestra el gráfico
    plt.show()
    
def victimas_mensuales(data):
    '''
    Crea gráficos de puntos para la cantidad de víctimas de accidentes mensuales por año utilizando seaborn.

    Parameters:
        data (pandas.DataFrame): El DataFrame que contiene los datos de accidentes, con una columna 'Año'.

    Returns:
        None
    '''
    # Se obtiene una lista de años únicos
    lista_años = data['Año'].unique()

    # Se define el número de filas y columnas para la cuadrícula de subgráficos
    num_filas = 3
    num_columnas = 2

    # Se crea una figura con subgráficos en una cuadrícula de 2x3
    fig, ejes = plt.subplots(num_filas, num_columnas, figsize=(14, 8))

    # Se itera a través de los años y crea un gráfico por año
    for indice, anio in enumerate(lista_años):
        fila_actual = indice // num_columnas
        columna_actual = indice % num_columnas
        
        # Se filtran los datos para el año actual y agrupa por mes
        datos_mensuales = (data[data['Año'] == anio]
                        .groupby('Mes')
                        .agg({'Cantidad de victimas':'sum'}))
        
        # Se configura el subgráfico actual
        ax_actual = ejes[fila_actual, columna_actual]
        sns.lineplot(data=datos_mensuales, ax=ax_actual, marker='o')
        ax_actual.set_title('Año ' + str(anio)) ; ax_actual.set_xlabel('Mes') ; ax_actual.set_ylabel('Cantidad de Víctimas')
        ax_actual.legend_ = None
        
    # Se muestra y acomoda el gráfico
    plt.tight_layout()
    plt.show()

def total_victimas_por_mes(dataframe):
    '''
    Crea un gráfico de barras que muestra la cantidad de víctimas de accidentes por mes utilizando plotly.

    Parameters:
        dataframe (pandas.DataFrame): El DataFrame que contiene los datos de accidentes con una columna 'Mes'.

    Returns:
        None
    '''
    # Se agrupa por la cantidad de víctimas por mes
    datos_agrupados = dataframe.groupby('Mes').agg({'Cantidad de victimas':'sum'}).reset_index()
    
    # Se grafica
    fig = px.bar(datos_agrupados, x='Mes', y='Cantidad de victimas', title='Cantidad de víctimas por Mes')
    fig.update_layout(xaxis_title='Mes', yaxis_title='Cantidad de Accidentes', xaxis = dict(tickmode = 'linear', dtick = 1))
    fig.show()
    
    # Se imprime resumen
    print(f'Menor cantidad de victimas: {datos_agrupados.min()[1]} victimas')
    print(f'Mayor cantidad de victimas: {datos_agrupados.max()[1]} victimas')
    
def accidentes_por_tipo_dia(datos):
    '''
    Genera un gráfico de barras que muestra la cantidad de accidentes por tipo de dia (semana o fin de semana) utilizando plotly.

    Parameters:
        datos: El conjunto de datos de accidentes.

    Returns:
        Un gráfico de barras.
    '''
    # Se convierte la columna 'Fecha' a tipo de dato datetime
    datos['Fecha'] = pd.to_datetime(datos['Fecha'])
    
    # Se extrae el dia de la semana (0 = lunes, 6 = domingo)
    datos['Dia de la semana'] = datos['Fecha'].dt.dayofweek
    
    # Se crea una columna 'Tipo de dia' para diferenciar entre semana y fin de semana
    datos['Tipo de dia'] = datos['Dia de la semana'].apply(lambda x: 'Fin de Semana' if x >= 5 else 'Semana')
    
    # Se cuenta la cantidad de accidentes por tipo de dia
    datos_agrupados = datos['Tipo de dia'].value_counts().reset_index()
    datos_agrupados.columns = ['Tipo de dia', 'Cantidad de accidentes']
    
    # Se crea el gráfico de barras
    fig = px.bar(datos_agrupados, x='Tipo de dia', y='Cantidad de accidentes', title='Cantidad de accidentes por tipo de dia')
    fig.update_layout(xaxis_title='Tipo de dia', yaxis_title='Cantidad de Accidentes')
    fig.show()
    
    # Se imprime resumen
    print(f'Dia con menor cantidad de accidentes: {datos_agrupados.min()[1]} accidentes')
    print(f'Dia con mayor cantidad de accidentes: {datos_agrupados.max()[1]} accidentes')

def resumen_victimas_sexo(datos):
    '''
    Genera un resumen de la cantidad de víctimas por sexo, rol y tipo de vehículo en un accidente de tráfico utilizando plotly.

    Parameters:
        datos (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    '''
    # Gráfico 1: Sexo
    fig1 = px.histogram(datos, x='Sexo', title='Cantidad de víctimas por sexo')
    fig1.update_layout(xaxis_title='Sexo', yaxis_title='Cantidad de víctimas')
    fig1.show()

    # Gráfico 2: Rol
    fig2 = px.histogram(datos, x='Rol', color='Sexo', title='Cantidad de víctimas por rol')
    fig2.update_layout(xaxis_title='Rol', yaxis_title='Cantidad de víctimas')
    fig2.show()

    # Gráfico 3: Tipo de vehiculo
    fig3 = px.histogram(datos, x='Victima', color='Sexo', title='Cantidad de víctimas por tipo de vehículo')
    fig3.update_layout(xaxis_title='Tipo de vehiculo', yaxis_title='Cantidad de victimas')
    fig3.show()

def grafico_edades(datos):
    '''
    Genera un gráfico con un histograma y un boxplot que muestran la distribución de la edad de los involucrados en los accidentes.

    Parameters:
        datos: El conjunto de datos de accidentes.

    Returns:
        Un gráfico con un histograma y un boxplot.
    '''
    # Se grafica el histograma de la edad
    fig_hist = px.histogram(datos, x='Edad', nbins=20, title='Histograma de Edades')
    fig_hist.show()

    # Se grafica el boxplot de la edad
    fig_box = px.box(datos, x='Edad', title='Boxplot de Edades')
    fig_box.show()

def accidentes_por_anio_y_genero(datos):
    '''
    Genera un gráfico de barras que muestra la cantidad total de víctimas por año y género.

    Parameters:
        datos: El conjunto de datos de accidentes.

    Returns:
        Un gráfico de barras.
    '''
    # Se crea el gráfico de barras
    fig = px.bar(datos, x='Año', y='Cantidad de victimas', color='Sexo', barmode='group',
                 title='Cantidad Total de Accidentes por Año y Género')
    fig.update_layout(yaxis_title='Cantidad Total de victimas')
    fig.show()

def hedges_g_por_año(df):
    '''
    Calcula el tamaño del efecto de Hedges g para dos grupos para los años del DataFrame y genera un gráfico con plotly.express.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        El tamaño del efecto de Hedges g y el gráfico correspondiente.
    '''
    def hedges_g(grupo1, grupo2):
        n1, n2 = len(grupo1), len(grupo2)
        mean1, mean2 = np.mean(grupo1), np.mean(grupo2)
        sd1, sd2 = np.std(grupo1, ddof=1), np.std(grupo2, ddof=1)
        
        # Calcula la desviación estándar agrupada
        sd_pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
        
        # Calcula el tamaño del efecto de Hedges g
        g = (mean1 - mean2) / sd_pooled
        
        # Ajuste para tamaños de muestra pequeños
        correction_factor = 1 - (3 / (4*(n1 + n2) - 9))
        g_adjusted = g * correction_factor
        
        return g_adjusted

    # Se obtienen los años del conjunto de datos
    años_unicos = df['Año'].unique()
    # Se crea una lista vacía para guardar los valores de Hedges g
    hedges_lista = []
    # Se itera por los años y se guarda Hedges g para cada grupo
    for a in años_unicos:
        grupo1 = df[((df['Sexo'] == 'MASCULINO') & (df['Año'] == a))]['Edad']
        grupo2 = df[((df['Sexo'] == 'FEMENINO') & (df['Año'] == a))]['Edad']
        g = hedges_g(grupo1, grupo2)
        hedges_lista.append(g)

    # Se crea un DataFrame
    hedges_df = pd.DataFrame()
    hedges_df['Año'] = años_unicos
    hedges_df['Estadístico de Hedges'] = hedges_lista
    
    # Se genera el gráfico con plotly.express
    fig = px.bar(hedges_df, x='Año', y='Estadístico de Hedges', title='Estadístico de Hedges g por Año')
    fig.show()

    return hedges_df

def hedges_g_total(df):
    '''
    Calcula el tamaño del efecto de Hedges g para dos grupos para la totalidad del DataFrame.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        El tamaño del efecto de Hedges g para la totalidad de los datos.
    '''
    def hedges_g(grupo1, grupo2):
        n1, n2 = len(grupo1), len(grupo2)
        mean1, mean2 = np.mean(grupo1), np.mean(grupo2)
        sd1, sd2 = np.std(grupo1, ddof=1), np.std(grupo2, ddof=1)
        
        # Calcula la desviación estándar agrupada
        sd_pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
        
        # Calcula el tamaño del efecto de Hedges g
        g = (mean1 - mean2) / sd_pooled
        
        # Ajuste para tamaños de muestra pequeños
        correction_factor = 1 - (3 / (4*(n1 + n2) - 9))
        g_adjusted = g * correction_factor
        
        return g_adjusted

    # Separa los grupos por género para la totalidad de los datos
    grupo1 = df[df['Sexo'] == 'MASCULINO']['Edad']
    grupo2 = df[df['Sexo'] == 'FEMENINO']['Edad']
    
    # Calcula Hedges g para la totalidad de los datos
    g_total = hedges_g(grupo1, grupo2)

    return g_total

def victimas_edad_y_rol(df):
    '''
    Genera un gráfico de caja de la distribución de la edad de las víctimas por rol utilizando plotly.express.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    '''
    fig = px.box(df, y='Rol', x='Edad', title='Distribución de la Edad de las Víctimas por Rol')
    fig.show()
    
def participantes_victimas_accidentes(df):
    '''
    Genera un gráfico interactivo de barras que muestra la cantidad total de víctimas por número de participantes en un accidente de tráfico.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    '''
    # Calcula la cantidad total de víctimas por número de participantes
    conteo_victimas = df.groupby('Participantes')['Victima'].count().reset_index()
    conteo_victimas = conteo_victimas.rename(columns={'Victima': 'Cantidad de victimas'})
    
    # Crea el gráfico interactivo de barras con plotly.express
    fig = px.bar(conteo_victimas, x='Participantes', y='Cantidad de victimas',
                 title='Resumen de victimas por Número de Participantes en Accidentes de Tráfico')
    fig.update_layout(xaxis_title='Número de Participantes', yaxis_title='Cantidad de victimas')
    fig.show()
    
def participantes_accidentes_calle_cruce(datos_accidentes):
    '''
    Genera un gráfico de barras horizontal que muestra la cantidad total de víctimas por tipo de calle y presencia de cruce.

    Parameters:
        datos_accidentes (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    '''
    # Agrupa los datos por 'Tipo de calle' y suma la 'Cantidad de victimas'
    total_victimas_calle = datos_accidentes.groupby('Tipo de calle')['Cantidad de victimas'].sum().reset_index()

    # Gráfico de barras horizontal para el total por tipo de calle
    fig_tipo_calle = px.bar(total_victimas_calle, x='Cantidad de victimas', y='Tipo de calle', orientation='h',
                            title='Total de Víctimas por Tipo de Calle')
    fig_tipo_calle.update_layout(xaxis_title='Total de Víctimas', yaxis_title='Tipo de Calle')
    fig_tipo_calle.show()

    # Agrupa los datos por 'Cruce' y suma la 'Cantidad de victimas'
    total_victimas_cruce = datos_accidentes.groupby('Cruce')['Cantidad de victimas'].sum().reset_index()

    # Gráfico de barras horizontal para el total por presencia de cruce
    fig_cruce = px.bar(total_victimas_cruce, x='Cantidad de victimas', y='Cruce', orientation='h',
                       title='Total de Víctimas en Cruces')
    fig_cruce.update_layout(xaxis_title='Total de Víctimas', yaxis_title='Presencia de Cruce')
    fig_cruce.show()
    
def acusados_en_accidentes(datos):
    '''
    Genera un gráfico de barras horizontal que muestra la cantidad total de acusados en accidentes de tráfico.

    Parameters:
        datos (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    '''
    # Agrupa los datos por 'Acusado' y cuenta las ocurrencias
    conteo_acusados = datos['Acusado'].value_counts().reset_index()
    conteo_acusados.columns = ['Acusado', 'Total']

    # Crea el gráfico de barras horizontal con plotly.express
    fig = px.bar(conteo_acusados, x='Total', y='Acusado', orientation='h',
                 title='Cantidad Total de Acusados en Accidentes de Tráfico')
    fig.update_layout(xaxis_title='Cantidad Total de Acusados', yaxis_title='Acusado')
    fig.show()

def convertir_columnas_a_numero(df):
    # Detectar las columnas que contienen 'Año' en el nombre y son de tipo string
    columnas_a_convertir = [columna for columna in df.columns if 'Año' in columna and df[columna].dtype == 'object']
    
    # Convertir las columnas detectadas
    for columna in columnas_a_convertir:
        df[columna] = df[columna].str.replace('.', '').astype(int)
    return df

def interpolacion_lineal(df):
    # Crear un rango de años entre 2016 y 2021
    rango_anios = list(range(2016, 2022))

    # Interpolar los valores de población para los años entre 2015 y 2020
    for anio in rango_anios:
        if anio not in df.columns:  # Verificar si el año ya está en el DataFrame
            # Interpolación lineal para estimar la población
            df[anio] = df['Año 2015'] + ((df['Año 2020'] - df['Año 2015']) / 5) * (anio - 2015)

    # Seleccionar solo las columnas de interés
    columnas_interes = ['Jurisdiccion'] + rango_anios
    df_poblacion_anio = df[columnas_interes]
    
    return df_poblacion_anio