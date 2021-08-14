
########################### TRABAJO FIN DE MÁSTER ########################

############################## PARTE 1 : PREPROCESADO ###################### 

########## DATASET INCENDIOS COMBINADO CON GALICIA  #########################

############################################################################### 


# ESTRUCTURA:

# 1. IMPORTACIÓN DE LIBRERÍAS

# 2. CREACIÓN DE DIRECTORIO DE TRABAJO E IMPORTACIÓN DE LOS DATOS 

# 3. DESCRIPCIÓN GENERAL DEL DATASET : INSPECCIÓN ANALÍTICA Y VISUAL

# 4. ESTUDIO Y TRATAMIENTO DE OUTLIERS

# 5. ESTUDIO Y TRATAMIENTO DE VALORES PERDIDOS

# 6. TRANSFORMACIÓN DE VARIABLES

# 7. CREACIÓN DE VARIABLES


##################################################################################
# 1. IMPORTACIÓN DE LIBRERÍAS
##################################################################################


# Vamos a importar las librerías que utilizaremos para la limpieza y el tratamiento de los datos y añadimos una nota breve para utilidad / finalidad de cada una.

# PONERLAS A MEDIDA QUE NOS LAS VAN PIDIENDO LAS FUNCIONES...

# Poner una breve descripción de cada una de las librerías para ver su finalidad

if (!"kableExtra" %in% installed.packages()) install.packages("kableExtra")
if (!"purrr" %in% installed.packages()) install.packages("purrr")
if (!"questionr" %in% installed.packages()) install.packages("questionr")
if (!"psych" %in% installed.packages()) install.packages("psych")
if (!"car" %in% installed.packages()) install.packages("car")
if (!"corrplot" %in% installed.packages()) install.packages("corrplot")
if (!"dplyr" %in% installed.packages()) install.packages("dplyr")
if (!"tidyverse" %in% installed.packages()) install.packages("tidyverse")
if (!"ggplot2" %in% installed.packages()) install.packages("ggplot2")
if (!"lubridate" %in% installed.packages()) install.packages("lubridate")
if (!"mice" %in% installed.packages()) install.packages("mice")
if (!"readr" %in% installed.packages()) install.packages("readr")


# Librerías:

library(kableExtra)
library(purrr)
library(questionr)
library(psych)
library(car)
library(corrplot)
library(dplyr)
library(tidyverse) # for data transformation 
library(ggplot2) # to plot beutiful grapt
library(lubridate) # to handle dates in elegant way
library(mice) # to check missing data
library(readr) # librería para leer datos


##################################################################################
# 2. CREACIÓN DE DIRECTORIO DE TRABAJO E IMPORTACIÓN DE LOS DATOS
##################################################################################



# Tenemos dos alternativas para la lectura de los datos: fijando un directorio de trabajo o leyendo los dataset desde una ubicación web

# Fijamos un directorio de trabajo:

setwd("C:/Users/Alicia López Machado/Desktop/Máster Big Data & Data Science/TFM")

# Leemos los dos dataset

#datos <- read.csv("DATASET TFM_v2_12072021.csv")

datos_galicia <- read.csv("Galicia_definitivo.csv")

# Lectura de datos desde github - DESCOMENTAR CON ####

library (readr) # librería para leer datos
url = "https://raw.githubusercontent.com/TFM123456/Datos/main/Galicia_definitivo.csv"  # URL actualizada 12/08/2021- debería funcionar para todos
# url2 = "https://github.com/TFM123456/Big_Data_and_Data_Science_UCM/blob/main/Galicia_definitivo.csv"
datos_galicia = as.data.frame(read.csv(url(url)))


url1 = "https://raw.githubusercontent.com/TFM123456/Datos/main/DATASET%20TFM_v2_12072021.csv"  # URL actualizada 12/08/2021- debería funcionar para todos
datos = as.data.frame(read.csv(url(url1)))



##################################################################################
# 3. DESCRIPCIÓN GENERAL DEL DATASET : INSPECCIÓN ANALÍTICA Y VISUAL
##################################################################################


# En este script tenemos contamos con 2 datasets. El primero de ellos, el de "datos" es aplicable al conjunto del territorio español, ya que las variables
#recogen información para todas las comunidades autónomas. Sobre este dataset,tenemos que decir que es compuesto, es decir, que se ha generado a partir de unos
# ya existente : xxxxxxxxxxxxxxxxxxxx al que hemos unido otro con información sobre variables mayoritariamente geográficas, para poder en un futuro, crear modelos
# más realistas y tener predicciones de mayor calidad, o con un r cuadrado mayor, entre otras razones.

# El dataset en el que se va a fundamentar el trabajo fin de máster es el de datos_galicia, que es un subset del dataset anterior. Hemos optado por trabajar con 
# la información recogida para Galicia, ya que es la Comunidad Autónoma en la que hemos comprobado que la disponibilidad de la información era mayor: ya sea por
# la mayor recogida de datos o la mayor incidencia del problema de incendios forestales.

# El preprocesado va a partir de el dataset_galicia, que se irá viendo modificado.


##################################################################################
#3.1) Exploración inicial y tipo de datos / variables y transformación 
##################################################################################

str(datos_galicia)
length(names(datos_galicia))

# Tenemos un total de 33 variables, de las cuales 14 están en el formato de character, 16 en formato de entero y 3 en formato numérico.
# En un primer vistazo vemos que el tipo de formato de estas variables lo tendríamos que transformar, para poder trabajar mejor con ellas.
# En este respecto, nuestras propuestas son las siguientes:

# -> Transformar la variable de fecha en el formato date

datos_galicia[,c(4)] = as.Date(datos_galicia$fecha, format ="%d/%m/%Y")

# -> Transformar las variables geográficas y geológicas en numéricas

datos_galicia[,c(23:33)] <- lapply(datos_galicia[,c(23:33)], as.numeric)

# Podemos seguir transformando variables, pero como van a ir acompañadas de más modificaciones, como cambios en sus valores, lo dejaremos para más adelante.


# -> También hemos considerado eliminar las variables de id concatenado, latlng_explicit y causa_supuesta, porque por la descripción de su contenido
# su papel o relevancia en nuestro análisis va a ser muy próximo a nulo.

datos_galicia[,c("ï..Concat")] = NULL
datos_galicia[,c("latlng_explicit")] = NULL
datos_galicia[,c("causa_supuesta")] = NULL
datos_galicia[,c("causa_desc")] = NULL # Eliminamos la variable de la descripción de las causas para no duplicar contenido
datos_galicia[,c("idcomunidad")] = NULL


# Comprobamos que se han eliminado correctamente estas columnas

length(names(datos_galicia))

# Después de realizar las primeras transformaciones a los tipos de datos sin alterar su contenido, hacemos un análisis rápido del dataset y comprobar
# la no duplicidad de ids.

length(unique(datos_galicia$id)) # No tenemos valores duplicados porque el número total de valores únicos es el mismo que el número de filas

summary(datos_galicia)  # Vemos que aproximadamente la mitad de las variables tienen valores perdidos, que analizaremos luego.


#head(datos_galicia)
dim(datos_galicia) 
#tail(datos_galicia)
glimpse(datos_galicia)
#psych::describe(Filter(is.numeric, datos_galicia))  usar para hacer un subset después de transformar

# Revisamos de nuevo las numéricas para ver si alguna más podría ser transformada en factor,y realizaremos esta transformación en el apartado siguiente

head(Filter(is.numeric, datos_galicia)) 


# En la transformación de variables que haremos a continuación , cambiará el contenido de la variable y el tipo de datos.
# En algunos casos la realizaremos para mejorar la interpretabilidad del modelo y para evitar la pérdida de observaciones en 
# el tratamiento de valores extremos y de valores perdidos.


##################################################################################
#3.2) Exploración gráfica inicial - dejar esta exploraración para el final del preprocesado
##################################################################################

# Vamos a generar dos funciones que utilizaremos más adelante para la inspección gráfica superficial:


dfplot_box <- function(data.frame, p=2){
  #par(mfrow=c(p,p))
  df <- data.frame
  ln <- length(names(data.frame))
  pl<-list()
  for(i in 1:ln){
    if(is.factor(df[,i])){
      b<-barras_cual(df[,i],nombreEje = names(df)[i])
      #print(b)
    } else {
      
      b<-boxplot_cont(df[,i],nombreEje = names(df)[i])}
    
    pl[[i]]<-b
  }  
  
  return(pl)
}

# Diagrama de cajas para las variables cuantitativas 

boxplot_cont<-function(var,nombreEje){
  dataaux<-data.frame(variable=var)
  ggplot(dataaux,aes(y=variable))+
    geom_boxplot(aes(x=""),notch=TRUE, fill="orchid") +
    stat_summary(aes(x="") ,fun.y=mean, geom="point", shape=8) +
    xlab(nombreEje)+ theme_light()+ theme(axis.title.y = element_blank())
}



#listaGraf <- dfplot_box(datos_galicia) #Boxplots
#boxplot_numericas = dfplot_box(numericas_df)
#names(numericas_df)
#gridExtra::marrangeGrob(boxplot_numericas, nrow = 4, ncol = 5)

# Posible representación del histograma en el EDA en Python

warning = FALSE # ejecutar si no funciona alguno de los gráficos



##################################################################################
# 4. TRANSFORMACIÓN Y CREACIÓN DE NUEVAS VARIABLES
##################################################################################

# Vamos a hacer una revisión rápida del contenido de cada una de las variables con :

head(datos_galicia)
str(datos_galicia)

# Por lo observado, podría ser significativo realizar transformaciones en las variables de:

# - 1 )  A partir de la variable de id provincia , municipio y comunidad, vamos a crear una nueva columna categorizando dicha variable
# - 2 )  A partir de la variable de causa, vamos a categorizar la misma y a condensar la información de 2 categorías en una muy similar
# - 3 )  A partir de la variable de fecha, vamos a generar 3 nuevas columnas, una que haa referencia al año al que pertecenen, otra al mes y otra al trimestre
# - 4 )  A partir de las variables de el tiempo del incendio y del tiempo de extinción, nos gustaría transformar estos valores a la duración de horas y minutos para mejorar su interpretabilidad
# - 5 )  Dicotomizar muertos heridos
# - 6 ) Hemos categorizado la variable de la dirección de viento para poder ubicar las observaciones en 8 posibles orientaciones en función de su dirección e investigar si existe alguna relación entre dicha dirección y las pérdidas generadas
# - 7 ) Dicotomizar precipitaciones

##################################################################################

#### 1 ) Transformación de idprovincia, municipio y comunidad- categorizar y clasificar en provincias

datos_galicia$idprovincia[datos_galicia$idprovincia == 15] <- "A Coruña"
datos_galicia$idprovincia[datos_galicia$idprovincia == 27] <- "Lugo"
datos_galicia$idprovincia[datos_galicia$idprovincia == 32] <- "Ourense"
datos_galicia$idprovincia[datos_galicia$idprovincia == 36] <- "Pontevedra"

datos_galicia$idmunicipio <- datos_galicia$municipio # y eliminamos municipio

datos_galicia$municipio = NULL

# En principio no vamos a eliminar ninguna variable por si pudiera ser utilizada en la modelización:

##################################################################################

#### 2 ) Transformación de causas a categórica reemplazando los números por valores

datos_galicia$causa <- replace(datos_galicia$causa , which(datos_galicia$causa==3),2)
datos_galicia$causa <- factor(datos_galicia$causa,
                              labels=c("rayo", "negligencia", "intencionado",
                                       "causa desconocida","fuego reproducido"))

datos_galicia$causa = as.character(datos_galicia$causa)

##################################################################################

#### 3 ) Crear nuevas columnas en fecha para : trimestre, años y meses 

# Partimos de la variable fecha adaptada al formato date en el primer apartado:

por_trimestres = datos_galicia %>% mutate(Trimestre = quarters(fecha))
por_meses = por_trimestres %>% mutate(Mes = months(fecha))
por_año = por_meses %>% mutate(Año = year(fecha))

datos_galicia = por_año

##################################################################################

#### 4 ) Tranformación de las variables de tiempo


# Primero vemos las primeras filas de esta variable y el formato inicial

head(datos_galicia$time_ctrl)
head(datos_galicia$time_ext)

# Creamos una variable puente para trabajar con ella y verificar que nos salen bien las transformaciones

totalMinutes = datos_galicia$time_ctrl

hour <- floor(totalMinutes / 60) # Redondea a la baja la hora
minute <- totalMinutes %% 60 * 0.01 # %% nos devuelve el resto de dividir entre 60
new_time <- hour + minute

datos_galicia$time_ctrl = new_time # Reemplazamos la variable que teníamos por la transformada

sort(datos_galicia$time_ctrl) # Comprobamos que al ordenar de forma creciente sí que funcióna
str(datos_galicia)

# Repetimos el mismo proceso con time_ext 


totalMinutes2 = datos_galicia$time_ext

hour2 <- floor(totalMinutes2 / 60) 
minute2 <- totalMinutes2 %% 60 * 0.01 
new_time2 <- hour + minute

datos_galicia$time_ext = new_time2 

head(datos_galicia$time_ext)
sort(datos_galicia$time_ext)

head(datos_galicia)
##################################################################################

#### 5 ) Dicotomizar muertos heridos


# En el caso de las variables de muertos y heridos, las vamos a dicotomizar asumiendo que en las observaciones donde
# no tenemos información recogida, el valor va a ser 0:

datos_galicia$muertos <- replace(datos_galicia$muertos , which(is.na(datos_galicia$muertos)),0)
datos_galicia$heridos <- replace(datos_galicia$heridos , which(is.na(datos_galicia$heridos)),0)


datos_galicia$muertos<-replace(datos_galicia$muertos, which(datos_galicia$muertos > 0), 1)
datos_galicia$heridos<-replace(datos_galicia$heridos, which(datos_galicia$heridos > 0), 1)

table(datos_galicia$muertos)
table(datos_galicia$heridos)

datos_galicia$muertos = as.factor(datos_galicia$muertos)
datos_galicia$heridos = as.factor(datos_galicia$heridos)


##################################################################################

#### 6 ) Categorizar la dirección del viento en función de 8 ejes


datos_galicia$DIR<-replace(datos_galicia$DIR, which((datos_galicia$DIR < 0)|(datos_galicia$DIR>36)), NA)

datos_galicia[,"DIR_VIENTO"] <- cut(datos_galicia$DIR,breaks = c(0, 2.25, 6.75, 11.25, 15.75, 20.25, 24.75, 29.25, 33.75, 36),
labels = c("N","NE","E","SE","S","SW","W","NW","N") ) 

datos_galicia$DIR = NULL

#Se realizan rupturas de 4.5º por cada dirección.
#0,36 = N -> Por tanto el Norte comprende entre 33.25 y 2.25
#4,5 = NE -> (2.25 - 6.75)
#9 = E -> (6.75 - 11.25)
#13,5 = SE (11.25 - - 15.75)
#18 = S (15.75 - 20.25)
#22,5 = SW -> (20.25 - 24.75)
#27 = W -> (24.75 - 29.25)
#31,5 = NW -> (29.25 - 33.75)

table(datos_galicia$DIR_VIENTO)
table(unique(datos_galicia$DIR_VIENTO))


##################################################################################

#### 7 ) Dicotomizar precipitaciones


# En el caso de las variables de muertos y heridos, las vamos a dicotomizar asumiendo que en las observaciones donde
# no tenemos información recogida, el valor va a ser 0:

datos_galicia$PRECIPITACION <- replace(datos_galicia$PRECIPITACION , which(is.na(datos_galicia$PRECIPITACION)),0)

datos_galicia$PRECIPITACION<-replace(datos_galicia$PRECIPITACION, which(datos_galicia$PRECIPITACION > 0), 1)

table(datos_galicia$PRECIPITACION)

datos_galicia$PRECIPITACION = as.factor(datos_galicia$PRECIPITACION)


##################################################################################


######## 4. ESTUDIO Y TRATAMIENTO DE OUTLIERS #######   
#####################################################


# Podemos utilizar algunas funciones de apoyo que hemos creado:


FindOutliers <- function(data) {
  Q1 <- quantile(data, .25,na.rm = TRUE)
  Q3 <- quantile(data, .75,na.rm = TRUE)
  IQR <- IQR(data,na.rm = TRUE) #Or use IQR(data)
  # we identify extreme outliers
  extreme.threshold.upper = (Q3 + 1.5*IQR)
  extreme.threshold.lower = (Q1 - 1.5*IQR)
  result <- which(data > extreme.threshold.upper | data < extreme.threshold.lower)
}

InciOutliers <- function(dataset,columna) {
  
  TotalOutliers = length(columna)
  Lengthdataset = dim(dataset)[1]
  Porc_outliers = (TotalOutliers / Lengthdataset )*100
  result <- Porc_outliers
}

RemoveOutliers <- function(dataset,column) {
  Q1 <- quantile(column, .25,na.rm = TRUE)
  Q3 <- quantile(column, .75,na.rm = TRUE)
  IQR <- IQR(column,na.rm = TRUE) #Or use IQR(data)
  
  # we identify extreme outliers
  
  result <- subset(dataset, column> (Q1 - 1.5*IQR) & column< (Q3 + 1.5*IQR))
}

# Vamos a crear un subset de variables numéricas para representar un primer boxplot


input<-as.data.frame(datos_galicia[,-(1)])
input<-as.data.frame(input[,-c(7,15)]) # Eliminamos perdidas - variable objetivo continua
row.names(input)<-datos_galicia$id

# Dentro de input vamos a hacer un subset para las numéricas y otro para las categóricas porque la identificación es distinta:

numericas_input <- input %>% select(is.numeric)
no_numericas_input <- input %>% select(!is.numeric)

length(names(numericas_input))
length(names(no_numericas_input))

names(numericas_input)

# Representamos el boxplot de este dataframe con el código que teníamos al principio del script:

listaGraf_input <- dfplot_box(numericas_input) #Boxplots
gridExtra::marrangeGrob(listaGraf_input, nrow = 4, ncol = 5)


# En la mayoría de variables es evidente que hay valores extremos y a raíz del análisis gráfico, recibirán tratamientos distintos:

outliers_superficie = FindOutliers(datos_galicia$superficie)
O_1 = InciOutliers(datos_galicia,outliers_superficie)

outliers_lat = FindOutliers(datos_galicia$lat)
O_2 = InciOutliers(datos_galicia,outliers_lat)

outliers_lng = FindOutliers(datos_galicia$lng)
O_3 = InciOutliers(datos_galicia,outliers_lng)

outliers_time_ctrl = FindOutliers(datos_galicia$time_ctrl)
O_4 = InciOutliers(datos_galicia,outliers_time_ctrl)

outliers_time_ext = FindOutliers(datos_galicia$time_ext)
O_5 = InciOutliers(datos_galicia,outliers_time_ext)

outliers_personal = FindOutliers(datos_galicia$personal)
O_6 = InciOutliers(datos_galicia,outliers_personal)

outliers_medios = FindOutliers(datos_galicia$medios)
O_7 = InciOutliers(datos_galicia,outliers_medios)

outliers_gastos = FindOutliers(datos_galicia$gastos)
O_8 = InciOutliers(datos_galicia,outliers_gastos)

outliers_ALTITUD = FindOutliers(datos_galicia$ALTITUD)
O_9 = InciOutliers(datos_galicia,outliers_ALTITUD)

outliers_TMEDIA = FindOutliers(datos_galicia$TMEDIA)
O_10 = InciOutliers(datos_galicia,outliers_TMEDIA)

outliers_TMIN = FindOutliers(datos_galicia$TMIN)
O_11 = InciOutliers(datos_galicia,outliers_TMIN)

outliers_TMAX = FindOutliers(datos_galicia$TMAX)
O_12 = InciOutliers(datos_galicia,outliers_TMAX)

outliers_VELMEDIA = FindOutliers(datos_galicia$VELMEDIA)
O_13 = InciOutliers(datos_galicia,outliers_VELMEDIA)

outliers_RACHA = FindOutliers(datos_galicia$RACHA)
O_14 = InciOutliers(datos_galicia,outliers_RACHA)

outliers_SOL = FindOutliers(datos_galicia$SOL)
O_15 = InciOutliers(datos_galicia,outliers_SOL)

outliers_PRESMAX = FindOutliers(datos_galicia$PRESMAX)
O_16 = InciOutliers(datos_galicia,outliers_PRESMAX)

outliers_PRESMIN = FindOutliers(datos_galicia$PRESMIN)
O_17 = InciOutliers(datos_galicia,outliers_PRESMIN)

outliers_Año = FindOutliers(datos_galicia$Año)
O_18 = InciOutliers(datos_galicia,outliers_Año)


variable <- c(names(numericas_input))
incidencia <- c(O_1,O_2,O_3,O_4,O_5,O_6,O_7,O_8,O_9,O_10,O_11,O_12,O_13,O_14,O_15,O_16,O_17,O_18)


incidencia_outliers <- data.frame(variable,
                           incidencia)


##PRIMERA PRUEBA
##################################################################################

#### 8 ) Categorizar los incendios por superficies


datos_galicia[,"superficie2"] <- cut(datos_galicia$superficie,breaks = c(0,100, 500, 1000,10000),
                                     labels = c("< 100 HA","100-500 HA","500-1000 HA","> 1000 HA") ) 
# Check 

length(which(datos_galicia$superficie2 == "> 1000 HA" ))
length(which(datos_galicia$superficie > 1000 ))
levels(datos_galicia$superficie2)
summary(datos_galicia$superficie2)

# Nos quedamos con la nueva columna de superficie categorizada

datos_galicia$superficie = datos_galicia$superficie2
datos_galicia$superficie2 = NULL

variable <- c(names(numericas_input[-(1)]))
incidencia <- c(O_2,O_3,O_4,O_5,O_6,O_7,O_8,O_9,O_10,O_11,O_12,O_13,O_14,O_15,O_16,O_17,O_18)


incidencia_outliers <- data.frame(variable,
                                  incidencia)

dim(datos_galicia)

#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$lat)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$lng)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$time_ctrl)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$time_ext)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$personal)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$medios)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$gastos)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$ALTITUD)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$TMEDIA)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$TMIN)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$TMAX)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$VELMEDIA)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$RACHA)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$SOL)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$PRESMAX)
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$PRESMIN)


#input<-as.data.frame(datos_galicia[,-(1)])
#input<-as.data.frame(input[,-c(7,15)]) # Eliminamos perdidas - variable objetivo continua
#row.names(input)<-datos_galicia$id

# Dentro de input vamos a hacer un subset para las numéricas y otro para las categóricas porque la identificación es distinta:

#numericas_input <- input %>% select(is.numeric)
#no_numericas_input <- input %>% select(!is.numeric)

#length(names(numericas_input))
#length(names(no_numericas_input))

#names(numericas_input)

# Representamos el boxplot de este dataframe con el código que teníamos al principio del script:

#listaGraf_input <- dfplot_box(numericas_input) #Boxplots
#gridExtra::marrangeGrob(listaGraf_input, nrow = 4, ncol = 5)

#dim(datos_galicia) # Se eliminan bastantes observaciones por lo que vamos a valorar categorizar otra variable

##################################################################################


#### 12 ) Categorizar PRESMAX y PRESMIN, pasarlas a binarias

summary(datos_galicia$PRESMAX)
datos_galicia[,"PRESMAX2"] <- cut(datos_galicia$PRESMAX,breaks = c(0,1000,10000),
                                  labels = c("INFERIOR A 1000","SUPERIOR A 1000") ) 

a = datos_galicia[,"PRESMAX2"]
b = datos_galicia[,"PRESMAX"]
s = as.data.frame(a)

s[,"b"] = b

datos_galicia$PRESMAX = datos_galicia[,"PRESMAX2"]

datos_galicia[,"PRESMAX2"] = NULL

datos_galicia[,"PRESMIN2"] <- cut(datos_galicia$PRESMIN,breaks = c(0,1000,10000),
                                  labels = c("INFERIOR A 1000","SUPERIOR A 1000") ) 

a = datos_galicia[,"PRESMIN2"]
b = datos_galicia[,"PRESMIN"]
s = as.data.frame(a)

s[,"b"] = b

datos_galicia$PRESMIN = datos_galicia[,"PRESMIN2"]

datos_galicia[,"PRESMIN2"] = NULL


input<-as.data.frame(datos_galicia[,-(1)])
input<-as.data.frame(input[,-c(7,15)]) # Eliminamos perdidas - variable objetivo continua
row.names(input)<-datos_galicia$id

numericas_input <- input %>% select(is.numeric)
names(numericas_input)
variable <- c(names(numericas_input))
incidencia <- c(O_2,O_3,O_4,O_5,O_6,O_7,O_8,O_9,O_10,O_11,O_12,O_13,O_14,O_15,O_18)


incidencia_outliers <- data.frame(variable,
                                  incidencia)

dim(datos_galicia) # 20545

#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$ALTITUD) # 17782
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$time_ctrl) # 15940 
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$time_ext) # 15482
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$VELMEDIA) # 14533
#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$medios) # 12959

# Solo con la eliminación de estas observaciones se nos ha ido más del 30% del dataset así que vamos a probar a categorizar alguna más

##################################################################################


#### 14 ) Tramificar altitud

head(datos_galicia$ALTITUD)

summary(datos_galicia$ALTITUD)
datos_galicia[,"ALTITUD2"] <- cut(datos_galicia$ALTITUD,breaks = c(0,80,100,125,150),
                                  labels = c("Inferior a 80","Entre 80-100","Entre 100-125","Superior a 125") ) 

datos_galicia$ALTITUD
a = datos_galicia[,"ALTITUD2"]
b = datos_galicia[,"ALTITUD"]
s = as.data.frame(a)

s[,"b"] = b


datos_galicia$ALTITUD2 = as.character(datos_galicia$ALTITUD2)

datos_galicia$ALTITUD2 = datos_galicia$ALTITUD2 %>% replace_na("NO INFO")

datos_galicia$ALTITUD = datos_galicia[,"ALTITUD2"]

datos_galicia$ALTITUD = as.factor(datos_galicia$ALTITUD)

datos_galicia[,"ALTITUD2"] = NULL

input<-as.data.frame(datos_galicia[,-(1)])
input<-as.data.frame(input[,-c(7,15)]) # Eliminamos perdidas - variable objetivo continua
row.names(input)<-datos_galicia$id

numericas_input <- input %>% select(is.numeric)
names(numericas_input)

variable <- c(names(numericas_input))
incidencia <- c(O_2,O_3,O_4,O_5,O_6,O_7,O_8,O_10,O_11,O_12,O_13,O_14,O_15,O_18)


incidencia_outliers <- data.frame(variable,
                                  incidencia)

incidencia_outliers

#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$time_ctrl)
#dim(datos_galicia) # 18434

#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$time_ext) 
#dim(datos_galicia) # 17907

#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$VELMEDIA) 
#dim(datos_galicia) # 16532

#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$medios) # 12959
#dim(datos_galicia) # 14520

#datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$personal)
#dim(datos_galicia) # 14175 


#input<-as.data.frame(datos_galicia[,-(1)])
#input<-as.data.frame(input[,-c(7,15)]) # Eliminamos perdidas - variable objetivo continua
#row.names(input)<-datos_galicia$id
#numericas_input <- input %>% select(is.numeric)
#listaGraf_input <- dfplot_box(numericas_input) #Boxplots
#gridExtra::marrangeGrob(listaGraf_input, nrow = 4, ncol = 5)

# Al representar con gastos distorsiona, así que vamos a categorizar esta variable


#### 9 ) Categorizar los gastos.

summary(datos_galicia$gastos)
head(datos_galicia$gastos)


datos_galicia[,"gastos2"] <- cut(datos_galicia$gastos,breaks = c(0,1000, 10000, 100000,1000000),
                                 labels = c("< 1K ","1K-10K","10K-100K","> 100K") ) 


a = datos_galicia[,"gastos2"]
b = datos_galicia[,"gastos"]
s = as.data.frame(a)

s[,"b"] = b

datos_galicia$gastos2 = as.character(datos_galicia$gastos2)

datos_galicia$gastos2 = datos_galicia$gastos2 %>% replace_na("NO INFO")

datos_galicia$gastos = datos_galicia$gastos2

datos_galicia$gastos = as.factor(datos_galicia$gastos)

datos_galicia$gastos2 = NULL


input<-as.data.frame(datos_galicia[,-(1)])
input<-as.data.frame(input[,-c(7,15)]) # Eliminamos perdidas - variable objetivo continua
row.names(input)<-datos_galicia$id
numericas_input <- input %>% select(is.numeric)

variable <- c(names(numericas_input))
incidencia <- c(O_2,O_3,O_4,O_5,O_6,O_7,O_10,O_11,O_12,O_13,O_14,O_15,O_18)


incidencia_outliers <- data.frame(variable,
                                  incidencia)

incidencia_outliers


datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$time_ctrl)
dim(datos_galicia) # 18434

datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$time_ext) 
dim(datos_galicia) # 17907

datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$VELMEDIA) 
dim(datos_galicia) # 16532

datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$medios) 
dim(datos_galicia) # 14520

datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$lat) 
dim(datos_galicia) # 14518


datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$personal) # 12959
dim(datos_galicia) # 14173 


datos_galicia = RemoveOutliers(datos_galicia,datos_galicia$RACHA) # 12959
dim(datos_galicia) # 14066



input<-as.data.frame(datos_galicia[,-(1)])
input<-as.data.frame(input[,-c(7,15)]) # Eliminamos perdidas - variable objetivo continua
row.names(input)<-datos_galicia$id
numericas_input <- input %>% select(is.numeric)
listaGraf_input <- dfplot_box(numericas_input) #Boxplots
gridExtra::marrangeGrob(listaGraf_input, nrow = 4, ncol = 5)


# Regeneramos las O´s


outliers_lat = FindOutliers(datos_galicia$lat)
O_2 = InciOutliers(datos_galicia,outliers_lat)

outliers_time_ctrl = FindOutliers(datos_galicia$time_ctrl)
O_4 = InciOutliers(datos_galicia,outliers_time_ctrl)

outliers_time_ext = FindOutliers(datos_galicia$time_ext)
O_5 = InciOutliers(datos_galicia,outliers_time_ext)

outliers_personal = FindOutliers(datos_galicia$personal)
O_6 = InciOutliers(datos_galicia,outliers_personal)

outliers_medios = FindOutliers(datos_galicia$medios)
O_7 = InciOutliers(datos_galicia,outliers_medios)

outliers_VELMEDIA = FindOutliers(datos_galicia$VELMEDIA)
O_13 = InciOutliers(datos_galicia,outliers_VELMEDIA)

outliers_RACHA = FindOutliers(datos_galicia$RACHA)
O_14 = InciOutliers(datos_galicia,outliers_RACHA)


incidencia <- c(O_2,O_3,O_4,O_5,O_6,O_7,O_10,O_11,O_12,O_13,O_14,O_15,O_18)


incidencia_outliers <- data.frame(variable,
                                  incidencia)

incidencia_outliers

# Como ahora el porcentaje de menores es mucho menor, vamos imputar por la media de las observaciones restantes:

# En estos casos, para generar una media más precisa eliminamos los valores extremos


capOutlier <- function(x){
  qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
  caps <- quantile(x, probs=c(.05, .95), na.rm = T)
  H <- 1.5 * IQR(x, na.rm = T)
  x[x < (qnt[1] - H)] <- caps[1]
  x[x > (qnt[2] + H)] <- caps[2]
  return(x)
}


dim(datos_galicia)

datos_galicia$time_ctrl=capOutlier(datos_galicia$time_ctrl)
datos_galicia$lng=capOutlier(datos_galicia$lng)
datos_galicia$time_ext=capOutlier(datos_galicia$time_ext)
datos_galicia$personal=capOutlier(datos_galicia$personal)
datos_galicia$TMEDIA=capOutlier(datos_galicia$TMEDIA)
datos_galicia$TMIN=capOutlier(datos_galicia$TMIN)
datos_galicia$TMAX=capOutlier(datos_galicia$TMAX)
datos_galicia$VELMEDIA=capOutlier(datos_galicia$VELMEDIA)

# Regeneramos las O´s

outliers_lng = FindOutliers(datos_galicia$lng)
O_3 = InciOutliers(datos_galicia,outliers_lng)

outliers_time_ctrl = FindOutliers(datos_galicia$time_ctrl)
O_4 = InciOutliers(datos_galicia,outliers_time_ctrl)

outliers_time_ext = FindOutliers(datos_galicia$time_ext)
O_5 = InciOutliers(datos_galicia,outliers_time_ext)

outliers_personal = FindOutliers(datos_galicia$personal)
O_6 = InciOutliers(datos_galicia,outliers_personal)

outliers_TMEDIA = FindOutliers(datos_galicia$TMEDIA)
O_10 = InciOutliers(datos_galicia,outliers_TMEDIA)

outliers_TMIN = FindOutliers(datos_galicia$TMIN)
O_11 = InciOutliers(datos_galicia,outliers_TMIN)

outliers_TMAX = FindOutliers(datos_galicia$TMAX)
O_12 = InciOutliers(datos_galicia,outliers_TMAX)

outliers_VELMEDIA = FindOutliers(datos_galicia$VELMEDIA)
O_13 = InciOutliers(datos_galicia,outliers_VELMEDIA)


incidencia <- c(O_2,O_3,O_4,O_5,O_6,O_7,O_10,O_11,O_12,O_13,O_14,O_15,O_18)


incidencia_outliers <- data.frame(variable,
                                  incidencia)

incidencia_outliers

input<-as.data.frame(datos_galicia[,-(1)])
input<-as.data.frame(input[,-c(7,15)]) # Eliminamos perdidas - variable objetivo continua
row.names(input)<-datos_galicia$id
numericas_input <- input %>% select(is.numeric)
listaGraf_input <- dfplot_box(numericas_input) #Boxplots
gridExtra::marrangeGrob(listaGraf_input, nrow = 4, ncol = 5)


##################################################################################
# Acciones que no hemos tomado
##################################################################################

#### 10 ) Categorizar time_ctrl.


# datos_galicia[,"time_ctrl2"] <- cut(datos_galicia$time_ctrl,breaks = c(0,5, 10, 100,10000),
#  labels = c("< 5 h","5-10 h","10-100 h","> 100 h") ) 
# a = datos_galicia[,"time_ctrl2"]
# b = datos_galicia[,"time_ctrl"]
# s = as.data.frame(a)

# s[,"b"] = b

# datos_galicia$time_ctrl = datos_galicia[,"time_ctrl2"]

# datos_galicia[,"time_ctrl2"] = NULL

##################################################################################

#### 11 ) Categorizar time_ext.

# summary(datos_galicia$time_ctrl)

# datos_galicia[,"time_ext2"] <- cut(datos_galicia$time_ext,breaks = c(0,5, 10, 100,10000),
#   labels = c("< 5 h","5-10 h","10-100 h","> 100 h") ) 

# a = datos_galicia[,"time_ext2"]
# b = datos_galicia[,"time_ext"]
# s = as.data.frame(a)

# s[,"b"] = b

# datos_galicia$time_ext = datos_galicia[,"time_ext2"]

# datos_galicia[,"time_ext2"] = NULL


##################################################################################




# Al igual que en el tratamiento de los valores perdidos, podremos tomar dos acciones: la eliminación de los valores extremos o el 
# reemplazo por la media o la media. Valoraremos en función de cada caso.



# Para analizar los outliers en cada una de las numéricas vamos a seguir el siguiente esquema:

# - Representar un boxplot inicial de las observaciones sin variar
# - Identificar los valores de outliers y ver la dimensión de los mismos respecto a la dimensión del dataset total para hacernos una idea de la incidencia
# - Eliminar los outliers 
# - Representar de nuevo el boxplot

# El tratamiento lo haremos en el datset de datos_galicia, ya que el de numéricas input lo usamos solo como referencia para
# ver los cambios en los diagramas de caja y bigotes



##################################################################################
# 5. ESTUDIO Y TRATAMIENTO DE VALORES PERDIDOS
##################################################################################

head(datos_galicia)
### -Localización- ###
######################

# Vamos a comprobar la cantidad de valores perdidos por cada una de las variables en valor absoluto y porcentual

total_perdidos = sapply(datos_galicia, function(datos_galicia) sum(is.na(datos_galicia))) # Cantidad de perdidos por variable
porcentaje_perdidos = 100*sapply(datos_galicia, function(datos_galicia) mean(is.na(datos_galicia))) # % de perdidos por variable

total_perdidos
porcentaje_perdidos

dim(datos_galicia)

# Las variables de latitud, longitud y las variables geográficas/ geológicas presentan un porcentaje muy bajo de valores perdidos
# por lo que el tratamiento más directo será eliminar las filas en las que no tengan información para esos registros.


head(datos_galicia)


# Por contra, el porcentaje de valores perdidos de las variables de muertos, heridos, gastos y pérdidas es mucho más significativo
# por lo que tendremos que dar un tratamiento diferente en este caso.

### -Tratamiento- ###
######################

# Para las variables geográficas vamos a eliminar los valores NA, es decir, quedarnos con las observaciones que no tienen estos valores perdidos:

#datos_galicia = datos_galicia[is.finite(datos_galicia$ALTITUD),]
#datos_galicia = datos_galicia[is.finite(datos_galicia$TMEDIA),]
#datos_galicia = datos_galicia[is.finite(datos_galicia$PRECIPITACION),]
#datos_galicia = datos_galicia[is.finite(datos_galicia$TMIN),]
#datos_galicia = datos_galicia[is.finite(datos_galicia$TMAX),]
datos_galicia = datos_galicia[is.finite(datos_galicia$DIR_VIENTO),]
#datos_galicia = datos_galicia[is.finite(datos_galicia$VELMEDIA),]
#datos_galicia = datos_galicia[is.finite(datos_galicia$RACHA),]
#datos_galicia = datos_galicia[is.finite(datos_galicia$SOL),]
datos_galicia = datos_galicia[is.finite(datos_galicia$PRESMAX),]
datos_galicia = datos_galicia[is.finite(datos_galicia$PRESMIN),]
#datos_galicia = datos_galicia[is.finite(datos_galicia$time_ext),]
#datos_galicia = datos_galicia[is.finite(datos_galicia$time_ctrl),]
datos_galicia = datos_galicia[is.finite(datos_galicia$perdidas),]

# Para las variables de latitud y longitud también vamos a eliminar los valores perdidos

#datos_galicia = datos_galicia[is.finite(datos_galicia$lat),]
#datos_galicia = datos_galicia[is.finite(datos_galicia$lng),]

# Ahora solo tenemos un número significativo de valores perdidos en las variables de : muertos, heridos,
# gastos y pérdidas.



# Para el caso de los gastos y de las pérdidas, vamos a analizar un poco más en profundidad el dataset para ver qué acciones debemos tomar:

# Tenemos dos opciones: 

# Eliminar las obervaciones o reemplazar por la media. 

#Vamos a comprobar la dispersión de la media de cada una de estas dos columnas 
#para ver si sería una acción adecuada reemplazar por este valor.

#coef_var <- function(x, na.rm = TRUE) {
# sd(x, na.rm=na.rm) / mean(x, na.rm=na.rm)
#}

#coef_var(datos_galicia$perdidas)
#mean(datos_galicia$perdidas, na.rm = TRUE)
#sd(datos_galicia$perdidas, na.rm = TRUE)
#var(datos_galicia$perdidas, na.rm = TRUE)


#coef_var(datos_galicia$gastos)
#mean(datos_galicia$gastos, na.rm = TRUE)
#sd(datos_galicia$gastos, na.rm = TRUE)
#var(datos_galicia$gastos, na.rm = TRUE)


# Atendiendo a las medidas de estadística descriptiva, no parece buena idea imputar por la media ya que la dispersión de las
# observaciones es elevada y podría inducir a errores. 


# En este caso, vamos a optar por la eliminación de los valores perdidos en la variables de pérdidas ( que será la que querremos predecir)
# y con la variable de gastos, parece razonable eliminarla en su totalidad, para no eliminar la información de otras observaciones
# si elimináramos esas líneas.


#datos_galicia[,c("prop_missings")] = NULL


# Comprobamos de nuevo que ninguna de las variables continúa con NA


total_perdidos = sapply(datos_galicia, function(datos_galicia) sum(is.na(datos_galicia))) # Cantidad de perdidos por variable
total_perdidos


head(datos_galicia)

write.csv(datos_galicia,'datos_galicia_limpio.csv')

