######### TRABAJO FIN DE MÁSTER ######


############ PARTE 1 : PREPROCESADO ######### 
############## #### #########################


# DATASET INCENDIOS COMBINADO CON GALICIA

# ESTRUCTURA:

# 1. CREACIÓN DE DIRECTORIO DE TRABAJO E IMPORTACIÓN DE LOS DATOS

# 2. IMPORTACIÓN DE LIBRERÍAS

# 3. DESCRIPCIÓN GENERAL DEL DATASET : INSPECCIÓN ANALÍTICA Y VISUAL

# 4. ESTUDIO Y TRATAMIENTO DE OUTLIERS

# 5. ESTUDIO Y TRATAMIENTO DE VALORES PERDIDOS

# 6. TRANSFORMACIÓN DE VARIABLES

# 7. CREACIÓN DE VARIABLES


##################################################################################
# 1. CREACIÓN DE DIRECTORIO DE TRABAJO E IMPORTACIÓN DE LOS DATOS
##################################################################################



# Tenemos dos alternativas para la lectura de los datos: fijando un directorio de trabajo o leyendo los dataset desde una ubicación web

# Fijamos un directorio de trabajo:

setwd("C:/Users/Alicia López Machado/Desktop/Máster Big Data & Data Science/TFM")

# Leemos los dos dataset

datos <- read.csv("DATASET TFM_v2_12072021.csv")

datos_galicia <- read.csv("Galicia_definitivo.csv")

## Meter directorio de Lena para la lectura de los datos:


######### (Lena) Leer el dataset directamente de GitHub -- porfi podéis poner aquí el código que funcionaba?

library (readr) # librería para leer datos
url = "https://raw.githubusercontent.com/TFM123456/Datos/main/Galicia_definitivo.csv"  # URL actualizada 12/08/2021- debería funcionar para todos
# url2 = "https://github.com/TFM123456/Big_Data_and_Data_Science_UCM/blob/main/Galicia_definitivo.csv"
datos_galicia = as.data.frame(read.csv(url(url)))


url1 = "https://raw.githubusercontent.com/TFM123456/Datos/main/DATASET%20TFM_v2_12072021.csv"  # URL actualizada 12/08/2021- debería funcionar para todos
datos = as.data.frame(read.csv(url(url1)))





##################################################################################
# 2. IMPORTACIÓN DE LIBRERÍAS
##################################################################################


# Vamos a importar las librerías que utilizaremos para la limpieza y el tratamiento de los datos y añadimos una nota breve para utilidad / finalidad de cada una.

# PONERLAS A MEDIDA QUE NOS LAS VAN PIDIENDO LAS FUNCIONES...

# Poner una breve descripción de cada una de las librerías para ver su finalidad

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

# Comprobamos que se han eliminado correctamente estas columnas

length(names(datos_galicia))

# Después de realizar las primeras transformaciones a los tipos de datos sin alterar su contenido, hacemos un análisis rápido del dataset y comprobar
# la no duplicidad de ids.

length(unique(datos_galicia$id)) # No tenemos valores duplicados porque el número total de valores únicos es el mismo que el número de filas

summary(datos_galicia)  # Vemos que aproximadamente la mitad de las variables tienen valores perdidos, que analizaremos luego.


#head(datos_galicia)
#dim(datos_galicia) 
#tail(datos_galicia)
glimpse(datos_galicia)
#psych::describe(Filter(is.numeric, datos_galicia))  usar para hacer un subset después de transformar

# Revisamos de nuevo las numéricas para ver si alguna más podría ser transformada en factor, pero en principio las mantenemos como están.

head(Filter(is.numeric, datos_galicia)) 

# En la transformación de variables que haremos a continuación , cambiará el contenido de la variable y el tipo de datos.



##################################################################################
#3.2) Exploración gráfica inicial - dejar esta exploraración para el final del preprocesado
##################################################################################

# Vamos a hacer un subset con la variables numéricas y otro con las categóricas, en las que deberían estar contenidas todas las variables


numericas_df <- datos_galicia %>% select(is.numeric)
categoricas_df <- datos_galicia %>% select(is.character)

#names(numericas_df)
#names(categoricas_df)

length(names(numericas_df)) 
length(names(categoricas_df))

#  Tenemos 28 numéricas y una categórica en esta altura, y junto a la del formato "date" hacen un total de 30, por lo que la división del dataset es correcta

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

listaGraf <- dfplot_box(datos_galicia) #Boxplots

boxplot_numericas = dfplot_box(numericas_df)
names(numericas_df)
gridExtra::marrangeGrob(boxplot_numericas, nrow = 4, ncol = 5)



# Ver si nos interesa la representación del histograma


listaHist<-dfplot_his(datos_galicia) #Histogramas
# pacman::p_load()

#gridExtra::marrangeGrob(listaHist, nrow = 3, ncol = 2)

#listaGraf <- dfplot_box(datos_galicia) #Boxplots
#listaGraf <- dfplot_box(datos_galicia[,c(1:4)]) #Boxplots

warning = FALSE # ejecutar si no funciona alguno de los gráficos


######## 4. ESTUDIO Y TRATAMIENTO DE OUTLIERS #######   
#####################################################

# Utilizar labels y levels para las variables categóricas

# Voy a indicar la variable objetivo contínua, que son las pérdidas a causa de los incendios.
# También indicaré el id, los valores atípicos y los perdidos sólo se gestionarán del resto de variables:

#Indico la variableObj, el ID y las Input 
# los atípicos y los missings se gestionan sólo de las input

varObjCont<-datos_galicia$perdidas
input<-as.data.frame(datos_galicia[,-(1)])

input<-as.data.frame(input[,-c(18)])

row.names(input)<-datos_galicia$id

# Dentro de input vamos a hacer un subset para las numéricas y otro para las categóricas porque la identificación es distinta:

numericas_input <- input %>% select(is.numeric)
categoricas_input <- input %>% select(is.character)



# Como hay algunas variables que no hemos categorizado todavía, tienen formato de numéricas pero en este caso no es relevante.

names(numericas_input)

numericas_input = numericas_input[,-c(4:10)]

# Representamos el boxplot de este dataframe con el código que teníamos al principio del script:


listaGraf_input <- dfplot_box(numericas_input) #Boxplots
gridExtra::marrangeGrob(listaGraf_input, nrow = 4, ncol = 5)
# En la mayoría de variables es evidente que hay valores extremos y a raíz del análisis gráfico, recibirán tratamientos distintos:



### 5.1 ) Tratamiento de numéricas- ###
######################


# Nuestro subset de referencia va a ser numericas_input


# Al igual que en el tratamiento de los valores perdidos, podremos tomar dos acciones: la eliminación de los valores extremos o el 
# reemplazo por la media o la media. Valoraremos en función de cada caso.

# Las variables que vamos a tratar en este apartado son:

capOutlier <- function(x){
  qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
  caps <- quantile(x, probs=c(.05, .95), na.rm = T)
  H <- 1.5 * IQR(x, na.rm = T)
  x[x < (qnt[1] - H)] <- caps[1]
  x[x > (qnt[2] + H)] <- caps[2]
  return(x)
}

# Representamos los outliers de las numéricas
boxplot(numericas_input)$out



# Para analizar los outliers en cada una de las numéricas vamos a seguir el siguiente esquema:

# - Representar un boxplot inicial de las observaciones sin variar
# - Identificar los valores de outliers y ver la dimensión de los mismos respecto a la dimensión del dataset total para hacernos una idea de la incidencia
# - Eliminar los outliers 
# - Representar de nuevo el boxplot

# Vamos a indentificar las observaciones que se encuentran entre los quartiles 1 y 3 y a utilizar como referencia 1,5 veces el rango intercuartilico

#find Q1, Q3, and interquartile range for values in column <- nombre de la columna ->

#Q1 <- quantile(numericas_input$nombre de la columna, .25,na.rm = TRUE)
#Q3 <- quantile(numericas_input$nombre de la columna, .75,na.rm = TRUE)
#IQR <- IQR(numericas_input$nombre de la columna,na.rm = TRUE)

#only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
#no_outliers <- subset(numericas_input, numericas_input$nombre de la columna> (Q1 - 1.5*IQR) & numericas_input$nombre de la columna< (Q3 + 1.5*IQR))

#view row and column count of new data frame
#dim(no_outliers) 

# no_outliers será el dataset que nos iremos quedando en cada caso que renombraremos al final

## superficie - imputamos porque no podemos perder tantas observaciones

boxplot(datos_galicia$superficie)

Q1 <- quantile(datos_galicia$superficie, .25,na.rm = TRUE)
Q3 <- quantile(datos_galicia$superficie, .75,na.rm = TRUE)
IQR <- IQR(datos_galicia$superficie,na.rm = TRUE)

no_outliers <- subset(datos_galicia, datos_galicia$superficie> (Q1 - 1.5*IQR) & datos_galicia$superficie< (Q3 + 1.5*IQR))
dim(no_outliers) 
boxplot(no_outliers$superficie)

## lat - aquí sí que es asumible eliminar observaciones

Q1 <- quantile(datos_galicia$lat, .25,na.rm = TRUE)
Q3 <- quantile(datos_galicia$lat, .75,na.rm = TRUE)
IQR <- IQR(datos_galicia$lat,na.rm = TRUE)

no_outliers <- subset(datos_galicia, datos_galicia$lat> (Q1 - 1.5*IQR) & datos_galicia$lat< (Q3 + 1.5*IQR))

dim(no_outliers) 

outlier = (numericas_input$lat> (Q1 - 1.5*IQR) & numericas_input$lat< (Q3 + 1.5*IQR))
length(no_outliers$lat)

boxplot(no_outliers$lat)

## lng # no hay outliers

boxplot(no_outliers$lng)

Q1 <- quantile(no_outliers$lng, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$lng, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$lng,na.rm = TRUE)
no_outliers <- subset(datos_galicia, no_outliers$lng> (Q1 - 1.5*IQR) & no_outliers$lng< (Q3 + 1.5*IQR))
dim(no_outliers) 

boxplot(no_outliers$lng)

## time_ctrl - se pierden muchas observaciones mejor reemplazar

boxplot(no_outliers$time_ctrl)

Q1 <- quantile(no_outliers$time_ctrl, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$time_ctrl, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$time_ctrl,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$time_ctrl> (Q1 - 1.5*IQR) & no_outliers$time_ctrl< (Q3 + 1.5*IQR))
dim(no_outliers) 

boxplot(no_outliers$time_ctrl)

## time_ext

boxplot(no_outliers$time_ext)

Q1 <- quantile(no_outliers$time_ext, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$time_ext, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$time_ext,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$time_ext> (Q1 - 1.5*IQR) & no_outliers$time_ext< (Q3 + 1.5*IQR))
dim(no_outliers) 

boxplot(no_outliers$time_ext)

## personal demasiadas observaciones, ver si nos interesa cambiar el rango intercuartilico

boxplot(no_outliers$personal)

Q1 <- quantile(no_outliers$personal, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$personal, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$personal,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$personal> (Q1 - 1.5*IQR) & no_outliers$personal< (Q3 + 1.5*IQR))
dim(no_outliers) 

a = length(datos_galicia$personal) - length(no_outliers$personal)

a

boxplot(no_outliers$personal)


## medios # se eliminan muchas observaciones

boxplot(no_outliers$medios)

Q1 <- quantile(no_outliers$medios, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$medios, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$medios,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$medios> (Q1 - 1.5*IQR) & no_outliers$medios< (Q3 + 1.5*IQR))
dim(no_outliers) 

a = length(datos_galicia$medios) - length(no_outliers$medios)

a

boxplot(no_outliers$personal)


## ALTITUD # se eliminan muchas observaciones


boxplot(no_outliers$ALTITUD)

Q1 <- quantile(no_outliers$ALTITUD, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$ALTITUD, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$ALTITUD,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$ALTITUD> (Q1 - 1.5*IQR) & no_outliers$ALTITUD< (Q3 + 1.5*IQR))
dim(no_outliers) 

a = length(datos_galicia$ALTITUD) - length(no_outliers$ALTITUD)

a

boxplot(no_outliers$ALTITUD)



## TMEDIA

boxplot(no_outliers$TMEDIA)

Q1 <- quantile(no_outliers$TMEDIA, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$TMEDIA, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$TMEDIA,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$TMEDIA> (Q1 - 1.5*IQR) & no_outliers$TMEDIA< (Q3 + 1.5*IQR))
dim(no_outliers) 

a = length(datos_galicia$TMEDIA) - length(no_outliers$TMEDIA)

a

boxplot(no_outliers$ALTITUD)

## PRECIPITACION -- ?

boxplot(no_outliers$PRECIPITACION)

Q1 <- quantile(no_outliers$PRECIPITACION, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$PRECIPITACION, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$PRECIPITACION,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$PRECIPITACION> (Q1 - 1.5*IQR) & no_outliers$PRECIPITACION< (Q3 + 1.5*IQR))
dim(no_outliers) 

a = length(datos_galicia$PRECIPITACION) - length(no_outliers$PRECIPITACION)

a

boxplot(no_outliers$PRECIPITACION)


## TMIN


boxplot(no_outliers$TMIN)

Q1 <- quantile(no_outliers$TMIN, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$TMIN, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$TMIN,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$TMIN> (Q1 - 1.5*IQR) & no_outliers$TMIN< (Q3 + 1.5*IQR))
dim(no_outliers) 

a = length(datos_galicia$TMIN) - length(no_outliers$TMIN)

boxplot(no_outliers$TMIN)

## TMAX

boxplot(no_outliers$TMAX)

Q1 <- quantile(no_outliers$TMAX, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$TMAX, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$TMAX,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$TMIN> (Q1 - 1.5*IQR) & no_outliers$TMIN< (Q3 + 1.5*IQR))
dim(no_outliers) 

a = length(datos_galicia$TMAX) - length(no_outliers$TMAX)

boxplot(no_outliers$TMAX)


## DIR

datos_galicia$DIR<-replace(datos_galicia$DIR, which((datos_galicia$DIR < 0)|(datos_galicia$DIR>36)), NA)

## VELMEDIA

boxplot(no_outliers$VELMEDIA)

Q1 <- quantile(no_outliers$VELMEDIA, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$VELMEDIA, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$VELMEDIA,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$TMIN> (Q1 - 1.5*IQR) & no_outliers$TMIN< (Q3 + 1.5*IQR))
dim(no_outliers) 

a = length(datos_galicia$VELMEDIA) - length(no_outliers$VELMEDIA)

boxplot(no_outliers$VELMEDIA)


## RACHA


boxplot(no_outliers$RACHA)

Q1 <- quantile(no_outliers$RACHA, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$RACHA, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$RACHA,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$RACHA> (Q1 - 1.5*IQR) & no_outliers$RACHA< (Q3 + 1.5*IQR))
dim(no_outliers) 

a = length(datos_galicia$RACHA) - length(no_outliers$RACHA)

boxplot(no_outliers$RACHA)


## PRESMAX


boxplot(no_outliers$PRESMAX)

Q1 <- quantile(no_outliers$PRESMAX, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$PRESMAX, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$PRESMAX,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$PRESMAX> (Q1 - 1.5*IQR) & no_outliers$PRESMAX< (Q3 + 1.5*IQR))
dim(no_outliers) 

a = length(datos_galicia$PRESMAX) - length(no_outliers$PRESMAX)

boxplot(no_outliers$PRESMAX)


## PRESMIN


boxplot(no_outliers$PRESMIN)

Q1 <- quantile(no_outliers$PRESMIN, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$PRESMIN, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$PRESMIN,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$PRESMIN> (Q1 - 1.5*IQR) & no_outliers$PRESMIN< (Q3 + 1.5*IQR))
dim(no_outliers) 

a = length(datos_galicia$PRESMIN) - length(no_outliers$PRESMIN)

boxplot(no_outliers$PRESMIN)


## SOL



boxplot(no_outliers$SOL)

Q1 <- quantile(no_outliers$SOL, .25,na.rm = TRUE)
Q3 <- quantile(no_outliers$SOL, .75,na.rm = TRUE)
IQR <- IQR(no_outliers$SOL,na.rm = TRUE)
no_outliers <- subset(no_outliers, no_outliers$SOL> (Q1 - 1.5*IQR) & no_outliers$SOL< (Q3 + 1.5*IQR))
dim(no_outliers) 

a = length(datos_galicia$SOL) - length(no_outliers$SOL)

boxplot(no_outliers$SOL)



# Código adicional de outliers



# Number of SDs to be considered an outlier
n=3

# Percent outliers in every column
numericas_input %>% 
  summarise_each(funs(sum(. > mean(.) + n*sd(.) | . < mean(.) - n*sd(.))/n()))


# Vamos a llevar a cabo un conteo de los valores que se consideran extremos según un consenso de dos criterios distintos. 
# En primer lugar, se distingue variable simétrica o posiblemente no, para aplicar *media + 3 sd* ó *mediana + 8 mad*, respectivamente. 
# Todas las medidas de dispersión basadas en la media o cuartiles son muy poco sensibles a la presencia de asimetría en la distribución,
#siendo por ello más fiables en este caso. 
#Por otro lado, aplicamos el clásico criterio del boxplot umbrales en *cuartil1 - 3IQR* y *cuartil3+ 3IQR*. 





# Para las variables de ALTITUD, TMAX, SOL, TMEDIA, DIR, PRESMAX, PRESMIN Y TMIN se puede ver gráficamente los outliers, así que vamos a eliminar esas observaciones:

#datos_galicia$ALTITUD<-replace(datos_galicia$ALTITUD, which((datos_galicia$ALTITUD < 0)|(datos_galicia$ALTITUD>200)), NA)
#datos_galicia$TMAX<-replace(datos_galicia$TMAX, which((datos_galicia$TMAX < 0)|(datos_galicia$TMAX>50)), NA)
#datos_galicia$SOL<-replace(datos_galicia$SOL, which(datos_galicia$SOL<5), NA)
#datos_galicia$TMEDIA<-replace(datos_galicia$TMEDIA, which(datos_galicia$TMEDIA<5), NA)
#datos_galicia$DIR<-replace(datos_galicia$DIR, which((datos_galicia$DIR < 0)|(datos_galicia$DIR>36)), NA)
#datos_galicia$PRESMAX<-replace(datos_galicia$PRESMAX, which(datos_galicia$PRESMAX < 250), NA)
#datos_galicia$PRESMIN<-replace(datos_galicia$PRESMIN, which(datos_galicia$PRESMIN < 250), NA)
#datos_galicia$TMIN<-replace(datos_galicia$TMIN, which((datos_galicia$TMIN < 0)|(datos_galicia$TMIN>50)), NA)                            


#cat_df <- datos_galicia %>% select(is.character)
#names(cat_df)

#summary(datos_galicia)

##### CÓDIGO PARA SUSTITUIR LOS OUTLIERS Y NO PERDER OBSERVACIONES  ###

# There is a better way to solve this problem. An outlier is not any point over the 95th percentile or below the 5th percentile. 
#Instead, an outlier is considered so if it is below the first quartile – 1.5·IQR or above third quartile + 1.5·IQR.

#### capOutlier <- function(x){
####   qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
####   caps <- quantile(x, probs=c(.05, .95), na.rm = T)
####   H <- 1.5 * IQR(x, na.rm = T)
####   x[x < (qnt[1] - H)] <- caps[1]
####  x[x > (qnt[2] + H)] <- caps[2]
####   return(x)
#### }
#### df$colName=capOutlier(df$colName)
#Do the above line over and over for all of the columns in your data frame

#### boxplot(datos_galicia$DIR)

#### datos_galicia$DIR =capOutlier(datos_galicia$DIR)

#### Código para enviar los atípicos a missings ## no funciona


# Modifico los atípicos como missings
#### input[,as.vector(which(sapply(
####   input, class)=="numeric"))]<-sapply(Filter(
####    is.numeric, input),function(x) atipicosAmissing(x)[[1]])





##################################################################################
# 5. ESTUDIO Y TRATAMIENTO DE VALORES PERDIDOS
##################################################################################


### -Localización- ###
######################

# Vamos a comprobar la cantidad de valores perdidos por cada una de las variables en valor absoluto y porcentual

total_perdidos = sapply(datos_galicia, function(datos_galicia) sum(is.na(datos_galicia))) # Cantidad de perdidos por variable
porcentaje_perdidos = 100*sapply(datos_galicia, function(datos_galicia) mean(is.na(datos_galicia))) # % de perdidos por variable

total_perdidos
porcentaje_perdidos


# Las variables de latitud, longitud y las variables geográficas/ geológicas presentan un porcentaje muy bajo de valores perdidos
# por lo que el tratamiento más directo será eliminar las filas en las que no tengan información para esos registros.


# Por contra, el porcentaje de valores perdidos de las variables de muertos, heridos, gastos y pérdidas es mucho más significativo
# por lo que tendremos que dar un tratamiento diferente en este caso.

### -Tratamiento- ###
######################

# Para las variables geográficas vamos a eliminar los valores NA, es decir, quedarnos con las observaciones que no tienen estos valores perdidos:

datos_galicia = datos_galicia[is.finite(datos_galicia$ALTITUD),]
datos_galicia = datos_galicia[is.finite(datos_galicia$TMEDIA),]
datos_galicia = datos_galicia[is.finite(datos_galicia$PRECIPITACION),]
datos_galicia = datos_galicia[is.finite(datos_galicia$TMIN),]
datos_galicia = datos_galicia[is.finite(datos_galicia$TMAX),]
datos_galicia = datos_galicia[is.finite(datos_galicia$DIR),]
datos_galicia = datos_galicia[is.finite(datos_galicia$VELMEDIA),]
datos_galicia = datos_galicia[is.finite(datos_galicia$RACHA),]
datos_galicia = datos_galicia[is.finite(datos_galicia$SOL),]
datos_galicia = datos_galicia[is.finite(datos_galicia$PRESMAX),]
datos_galicia = datos_galicia[is.finite(datos_galicia$PRESMIN),]
datos_galicia = datos_galicia[is.finite(datos_galicia$PRESMAX),]

# Para las variables de latitud y longitud también vamos a eliminar los valores perdidos

datos_galicia = datos_galicia[is.finite(datos_galicia$lat),]
datos_galicia = datos_galicia[is.finite(datos_galicia$lng),]

# Ahora solo tenemos un número significativo de valores perdidos en las variables de : muertos, heridos,
# gastos y pérdidas.

# En el caso de las variables de muertos y heridos, las vamos a dicotomizar asumiendo que en las observaciones donde
# no tenemos información recogida, el valor va a ser 0:

datos_galicia$muertos <- replace(datos_galicia$muertos , which(is.na(datos_galicia$muertos)),0)
datos_galicia$heridos <- replace(datos_galicia$heridos , which(is.na(datos_galicia$heridos)),0)

# Para el caso de los gastos y de las pérdidas, vamos a analizar un poco más en profundidad el dataset para ver qué acciones debemos tomar:

# Tenemos dos opciones: 

# Eliminar las obervaciones o reemplazar por la media. 

#Vamos a comprobar la dispersión de la media de cada una de estas dos columnas 
#para ver si sería una acción adecuada reemplazar por este valor.

coef_var <- function(x, na.rm = TRUE) {
  sd(x, na.rm=na.rm) / mean(x, na.rm=na.rm)
}

coef_var(datos_galicia$perdidas)
mean(datos_galicia$perdidas, na.rm = TRUE)
sd(datos_galicia$perdidas, na.rm = TRUE)
var(datos_galicia$perdidas, na.rm = TRUE)


coef_var(datos_galicia$gastos)
mean(datos_galicia$gastos, na.rm = TRUE)
sd(datos_galicia$gastos, na.rm = TRUE)
var(datos_galicia$gastos, na.rm = TRUE)


# Atendiendo a las medidas de estadística descriptiva, no parece buena idea imputar por la media ya que la dispersión de las
# observaciones es elevada y podría inducir a errores. 


# En este caso, vamos a optar por la eliminación de los valores perdidos en la variables de pérdidas ( que será la que querremos predecir)
# y con la variable de gastos, parece razonable eliminarla en su totalidad, para no eliminar la información de otras observaciones
# si elimináramos esas líneas.


datos_galicia = datos_galicia[is.finite(datos_galicia$perdidas),]
datos_galicia[,c("prop_missings")] = NULL
datos_galicia[,c("gastos")] = NULL

# Comprobamos de nuevo que ninguna de las variables continúa con NA


total_perdidos = sapply(datos_galicia, function(datos_galicia) sum(is.na(datos_galicia))) # Cantidad de perdidos por variable
total_perdidos


##################################################################################
# 6. TRANSFORMACIÓN Y CREACIÓN DE NUEVAS VARIABLES
##################################################################################

# Vamos a hacer una revisión rápida del contenido de cada una de las variables con :

head(datos_galicia)
str(datos_galicia)

# POr lo observado, podría ser significativo realizar transformaciones en las variables de:

# - 1 )  A partir de la variable de id provincia , municipio y comunidad, vamos a crear una nueva columna categorizando dicha variable
# - 2 )  A partir de la variable de causa, vamos a categorizar la misma y a condensar la información de 2 categorías en una muy similar
# - 3 )  A partir de la variable de fecha, vamos a generar 3 nuevas columnas, una que haa referencia al año al que pertecenen, otra al mes y otra al trimestre
# - 4 )  A partir de las variables de el tiempo del incendio y del tiempo de extinción, nos gustaría transformar estos valores a la duración de horas y minutos para mejorar su interpretabilidad
# - 5 )  A partir de la variable de causa, hemos sintetizado las posibles causa porque sino esta varaible pierde interpretabilidad
# - 6 ) Hemos categorizado la variable de la dirección de viento para poder ubicar las observaciones en 8 posibles orientaciones en función de su dirección e investigar si existe alguna relación entre dicha dirección y las pérdidas generadas


#dataset$perdidas <- replace(dataset$perdidas ,
#                           which(is.na(dataset$perdidas)),0)

##################################################################################

#### 1 ) Transformación de idprovincia, municipio y comunidad- categorizar y clasificar en provincias

datos_galicia$idprovincia[datos_galicia$idprovincia == 15] <- "A Coruña"
datos_galicia$idprovincia[datos_galicia$idprovincia == 27] <- "Lugo"
datos_galicia$idprovincia[datos_galicia$idprovincia == 32] <- "Ourense"
datos_galicia$idprovincia[datos_galicia$idprovincia == 36] <- "Pontevedra"
datos_galicia$idcomunidad[datos_galicia$idcomunidad == 3] <- "Galicia"

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
str(datos_galicia)

# MEJORAR -> CREAR LAS COLUMNAS Y AÑADIRLAS, NO ALTERANDO EL DATASET POR DENTRO

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

str(datos_galicia)

##################################################################################

#### 5 ) Sintetizar las categorías de la causa detallada










##################################################################################

#### 6 ) Categorizar la dirección del viento en función de 8 ejes


datos_galicia$DIR<-replace(datos_galicia$DIR, which((datos_galicia$DIR < 0)|(datos_galicia$DIR>36)), 0)

datos_galicia[,"viento"] <- cut(datos_galicia$DIR,breaks = c(0,4.5,9,13.5,18,22.5,27,31.5,36),
                                labels = c("Norte","Noreste","Este","Sudeste","Sur","Sudoeste","Oeste","Noroeste") ) 

#levels(datos_galicia$viento)

str(datos_galicia)



# Ver si se pueden sacar 4 categorías de la dirección del viento  ALEX - investigar - quitar NA primero


datos_galicia$DIR<-replace(datos_galicia$DIR, which((datos_galicia$DIR < 0)|(datos_galicia$DIR>36)), 0)



#datos_galicia[,"viento"] <- cut(datos_galicia$DIR,breaks = c(0,4.5,9,13.5,18,22.5,27,31.5,36),
#labels = c("A","B","C","D","E","F","G","H") ) 


# Código de Alex

#datos_galicia[,"viento"] <- cut(datos_galicia$DIR,breaks = c(0, 2.25, 6.75, 11.25, 15.75, 20.25, 24.75, 29.25, 33.75, 36),
                                labels = c("N","NE","E","SE","S","SW","W","NW","N") ) 

#Se realizan rupturas de 4.5º por cada dirección.
#0,36 = N -> Por tanto el Norte comprende entre 33.25 y 2.25
#4,5 = NE -> (2.25 - 6.75)
#9 = E -> (6.75 - 11.25)
#13,5 = SE (11.25 - - 15.75)
#18 = S (15.75 - 20.25)
#22,5 = SW -> (20.25 - 24.75)
#27 = W -> (24.75 - 29.25)
#31,5 = NW -> (29.25 - 33.75)

table(datos_galicia$viento)   





######### EXTRA ##########



# Buscar los códigos de colinealidad ANA 

corr_num<-corrplot(cor(numlab_df), use="pairwise", method = "ellipse",type = "upper") # Meter después de los NAs


