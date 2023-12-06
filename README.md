<h1 align="center">Proyecto Individual Nº2: Siniestros Viales</h1>



![Imagen Proyecto](png/Imagen%20Proyecto.jpg)

## Introducción
Este estudio examinó las fatalidades en incidentes de tránsito en la Ciudad Autónoma de Buenos Aires, utilizando una base de datos oficial del gobierno argentino que compila detalles relevantes. Se analizaron los datos de accidentes ocurridos entre 2016 y 2021, garantizando la confidencialidad de los individuos implicados.

Los accidentes de tránsito, que pueden resultar en daños o incluso la muerte, fueron el foco de este análisis.

Anualmente, aproximadamente 4.000 argentinos pierden la vida en estos accidentes, siendo la causa principal de muerte violenta en el país. Según el Sistema Nacional de Información Criminal, de 2018 a 2022, se contabilizaron 19.630 decesos por esta causa, lo que representa un promedio de 11 muertes diarias. La probabilidad de fallecer en un accidente de tránsito en Argentina es dos o tres veces mayor que en situaciones de violencia criminal.

En Buenos Aires, la alta densidad de tráfico y población hace que estos accidentes sean una preocupación significativa.

El análisis se realizó en Python, utilizando librerías como pandas, matplotlib , seaborn y ploty.express.

El proceso comenzó con una fase de ETL, donde se verificó la limpieza de los datos, seguido de un análisis exploratorio de datos que permitió obtener varias conclusiones. Posteriormente, se compararon con KPIs relevantes y se presentaron los hallazgos en un tablero de control.

## Objetivo

- Examinar minuciosamente los conjuntos de datos para comprender adecuadamente la problemática y extraer datos significativos.

- Desarrollar un tablero de control que exhiba los resultados del análisis y facilite una comunicación clara y eficaz de la información.

## Proceso ETL

Se ejecutó el procedimiento ETL (Extracción, Limpieza y Transformación) para los datos. Se observó que el conjunto de datos estaba en condiciones óptimas para su uso, presentando mínimos datos incompletos o ausentes y sin incidencias de registros duplicados. Se generaron diversas tablas para simplificar el análisis, como la clasificación por semestres, y se eliminó la columna “Altura” por su escasa relevancia informativa.

*El desarrollo del proceso (ETL), se puede ver en el siguiente link* [ETL_Hechos](https://github.com/RobertinoS/PI_DA_Siniestros_Viales/blob/main/01_ETL_Hechos.ipynb)
                                                                     [ETL_Victimas](https://github.com/RobertinoS/PI_DA_Siniestros_Viales/blob/main/02_ETL_Victimas.ipynb)

## Análisis Exploratorio de Datos (EDA)

Tras el ETL, se llevó a cabo un EDA para poder identificar patrones y/o comportamientos de los datos provenientes de las base de datos pertinente. A contuniacion se observan las siguientes conclusiones que se obtuvieron, seguido de una conclusion general.

- Fluctuaciones Temporales de Víctimas:
- Entre 2016 y 2021 se registraron 717 víctimas en total.

- Distribución de víctimas:
    - Diariamente: Mayor número de accidentes entre las 5 y las 7 de la mañana.
    - Semanalmente: El miércoles tiene la menor cantidad de víctimas (101) y el domingo la mayor (117).
    - Mensualmente: Diciembre tiene la menor cantidad (51) y julio la mayor (87).

- Análisis por Sexo y Rol
- Sexo de las Víctimas:
    - Mayor cantidad de víctimas: Masculino (551).
- Roles:
    - Mayor cantidad de víctimas masculinas son conductores (320).
    - Mayor cantidad de víctimas femeninas son peatones (103).

- Tipo de Vehículo:
    - Mayor cantidad de víctimas masculinas en motos (266).
    - Mayor cantidad de víctimas femeninas son peatones (103).

- Edades de las Víctimas
    - Mediana de edad general: 39 años.

- Tendencias:
    - Hasta 2019, tendencia creciente en edades de mujeres fallecidas, disminuyendo en 2020.
    - Leve disminución en fallecimientos de hombres hasta 2020.

- Distribución por Roles:
    - Conductores y pasajeros tienen mediana de edad alrededor de los 35 años.
    - Peatones: Mediana de 55 años, dispersión entre 1 y 95 años.
    - Ciclistas: Mediana de 42 años, dispersión entre 5 y 86 años.

- Análisis Específicos
    - Participantes en Accidentes: Peatón-Pasajero tienen más víctimas, seguido de Moto-Auto y Moto-Cargas.
    - Calles donde ocurren los Hechos: Avenidas y cruces son los lugares con mayor número de accidentes.
    - Acusados en Accidentes: Conductores de autos tienen mayor responsabilidad en los hechos registrados.sabilidad de los hechos registrados es de los Conductores de autos

- **Conclusion General**: Los accidentes de tráfico en CABA entre 2016 y 2021 dejaron 717 víctimas. Mayormente involucraron a Hombres, siendo conductores de motos la categoría más afectada. Los accidentes ocurren principalmente en las Mañanas, en Cruces de Avenidas. Las edades de las víctimas varían: Peatones suelen ser mayores. Los conductores de autos tienen mayor responsabilidad en estos incidentes. Ademas se puede concluir que la disminucion de siniestros viales en el primer semestre del año 2020, puede ser, debido a la pandemia por COVID-19, más que porque hayan mejorado las condiciones viales

*El desarrollo del proceso (EDA), se puede ver en el siguiente link* [EDA](https://github.com/RobertinoS/PI_DA_Siniestros_Viales/blob/main/03_EDA.ipynb)
## Web Scraping

Realize un Web Scraping para determinar la poblacion en CABA (2016-2021). Luego calcule la media de la misma para tomar dicho valor como poblacion total y poder realizar los KPIs posteriormente. El sitio web de donde extraje los datos es la siguiente: (https://www.ign.gob.ar/NuestrasActividades/Geografia/DatosArgentina/Poblacion2).
*El desarrollo del proceso (Web Scraping), se puede ver en el siguiente link* [Web Scraping](https://github.com/RobertinoS/PI_DA_Siniestros_Viales/blob/main/04_WebScraping.ipynb)

## KPIs

En este proyecto, se requirieron 2 KPIs (Indicadores clave de rendimiento), de los cuales se entregaron mismos y adicionalmente se agrego un tercer Kpi, el cual se  diseñó basándose en los datos recopilados en este estudio. A continuación se detallan los KPIs empleados.

- Reducción del 10% en la tasa de homicidios en siniestros viales de los últimos seis meses en CABA, comparada con la tasa del semestre anterior.

- Disminución del 7% en la cantidad de accidentes mortales de motociclistas en el último año en CABA, en relación al año anterior.

- Reducción del 10% en la cantidad de accidentes mortales de automoviles en el último año en CABA, en relación al año anterior.

Las tasas de mortalidad asociadas a incidentes viales son indicadores cruciales de la seguridad en el tránsito de una región. Estas tasas se calculan generalmente como el número de muertes por cada cierto número de habitantes o vehículos registrados. La disminución de estas tasas es un objetivo principal para mejorar la seguridad vial y resguardar la vida de los ciudadanos en el entorno urbano, es decir, en la ciudad. Estos datos fueron fundamentales en la definición de los KPIs.

## DASHBOARD

Por ultimo, elabore el Dashboard interactivo en Power Bi, en el cual se pueden observar conclusiones tanto del EDA, como tambien de los KPIs.
*El desarrollo del proceso (DASHBOARD), se puede ver en el siguiente link* [DASHBOARD](https://github.com/RobertinoS/PI_DA_Siniestros_Viales/blob/main/homicidios.pbix)

## Stack Tecnologico:
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![PowerBI]( https://img.shields.io/badge/PowerBI-F2C811?style=for-the-badge&logo=Power%20BI&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Markdown](https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor=white)
![Plotly Express](https://img.shields.io/badge/plotly_express-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
