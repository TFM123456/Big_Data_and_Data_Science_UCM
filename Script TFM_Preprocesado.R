#Hola guapetones
# PREPROCESADO TFM

# Fijamos un directorio de trabajo:
load(url(https://github.com/TFM123456/Big_Data_and_Data_Science_UCM/blob/main/Galicia_definitivo.csv))


# Leemos los dos dataset

datos <- read.csv("DATASET TFM_v2_12072021.csv")


#ÀstringsAsFactors=F, na.strings=c(NA,"NA"," NA")

datos_galicia <- read.csv("Galicia.csv")


#. 1. IMPORTACIÓN DE LIBRERÍAS - poner para qué es cada una

# Librerías:

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


#library(leaflet)
#install.packages("leaflet")
#install.packages("ggplot2")
#install.packages("sf")
#library(sf)

.

## Que comprobar?
## - Tipos de variables
## - Valores mal codificados
## - Valores fuera de rango
## - Variables nominales o Factores con categorías minoritarias
## - Outliers (incidencia y conversión a missing)
## - Missings (incidencia e imputación)




## 1) DESCRIPCIÓN GENERAL DEL DATASET,
#(explicar cómo hemos creado el dataset más completo)

# Sobre el dataset general + explicar que el otro es el mismo enfocado en Galicia

#1.1) Exploración inicial y tipo de datos -    transformación- conversión a variables dicotómicas



dfplot_box(datos) 
dfplot_his(datos)

str(datos_galicia) #No todas las categóricas están como factores

# eliminar la de concat
# id ver que no hay duplicados
# superficie nas + outliers
# fecha, agrupar por- trimestres, meses, años y crear columnas nuevas ¿?
# latitud nas, outliers y siempre positivo y pasar a formato coordenadas¿? - secundario
#longitud , nas, outliers y siempre negativo
# latlng_explicit : quitarla porque no aporta nada
# ids de sitios : pasar a caracter, nas, outliers - sirven para agrupar
# causa, categorizar, nas, outliers, colinealidad V de Cramer?
# causa supuesta, dicotómica, nas, mirar cuantos nas
# causa descripción, categorizar
# muertos y heridos dicotomica
# time , nas, outliers y pasar , transformarla ?¿
# personal, medios, gastos,perdidas, nas y outliers - tramificar 

# variables climáticas, espaciales, geográficas - pasar todas a numéricas

# altitud , nas outliers muchas por debajo del nivel del mar¿?
# t media outliers y na , mandar las del 99 a na , imputar por ?
# dirección del viento , tramificar¿?- en la parte de la creación de nuevas varibles, crear 4 categorías


str(datos_galicia)
summary(datos_galicia) # usar esta para comprobar que los datos se han transformado bien
#head(datos_galicia)
dim(datos_galicia) # usar este código para comprobar la selección de variables con las columans que nos quedamos
#tail(datos_galicia)
glimpse(datos_galicia)
psych::describe(Filter(is.numeric, datos_galicia)) # usar para hacer un subset después de transformar

# Cuento el número de valores diferentes para las numéricas para ver si alguna más podría ser transformada en factor:
sapply(Filter(is.numeric, datos),function(x) length(unique(x))) 

#1.2) Exploración gráfica - simple tipos de graficos

# conjunto de boxplot





# TRANSFORMACIÓN DE TIPOS DE VARIABLES



# Transformación de todas las geográficas a numéricas

names(datos_galicia)

datos_galicia[,c(23:33)] <- lapply(datos_galicia[,c(23:33)], as.numeric)  # REVISAR

# Transformación de long y lat

#datos_galicia[,c(5:6)] <- lapply(datos_galicia[,c(5:6)], )


# Transformación de idprovincia, municipio y comunidad- categorizar y clasificar en provincias

datos_galicia$idprovincia[datos_galicia$idprovincia == 15] <- "A Coruña"
datos_galicia$idprovincia[datos_galicia$idprovincia == 27] <- "Lugo"
datos_galicia$idprovincia[datos_galicia$idprovincia == 32] <- "Ourense"
datos_galicia$idprovincia[datos_galicia$idprovincia == 36] <- "Pontevedra"


datos_galicia$idcomunidad[datos_galicia$idcomunidad == 3] <- "Galicia"


datos_galicia$idmunicipio <- datos_galicia$municipio # y eliminamos municipio

datos_galicia$municipio = NULL

# Transformación de causas a char 


datos_galicia$causa <- replace(datos_galicia$causa , which(datos_galicia$causa==3),2)
datos_galicia$causa <- factor(datos_galicia$causa,
                        labels=c("rayo", "negligencia", "intencionado",
                                 "causa desconocida","fuego reproducido"))

table(datos_galicia$causa)

# dicotimizar causa supuesta

datos_galicia$causa_supuesta <- replace(datos_galicia$causa_supuesta , which(is.na(datos_galicia$causa_supuesta)),0)




## 2) VALORES PERDIDOS


datos_galicia$prop_missings<-apply(is.na(datos_galicia),1,mean)
# Por observaciÃ³n
summary(datos_galicia$prop_missings)
(prop_missingsVars<-apply(is.na(datos_galicia),2,mean))

datos_galicia$lng


# -- Localización -- #

any(is.na(datos_galicia)) # Me dice si en la variable alguna de las observaciones es NA
is.na(x) # Me devuelve todos los que son NA
which(is.na(datos_galicia)) # Indica la posición en la que están los valores perdidos
sum(is.na(datos_galicia))  # Cantidad de valores perdidos en el vector
mean(is.na(x)) # Porcentaje de valores NA en la variable, para ver si es viable eliminar filas

sapply(datos_galicia, function(x) sum(is.na(x))) # Cantidad de perdidos por variable
sapply(datos_galicia, function(x) mean(is.na(datos_galicia))) # % de perdidos por variable

# Significación de la variable antes de eliminarla




# -- Tratamiento de NA -- #

na.omit(x)  # Para eliminarlos
na.fail(x) # Lanza error si los encuentra
complete.cases(data)


#- Omisión de valores perdidos
#- Tratamiento / imputación

dataset$muertos <- replace(dataset$muertos , which(is.na(dataset$muertos)),0)
dataset$heridos <- replace(dataset$heridos , which(is.na(dataset$heridos)),0)
dataset$causa <- replace(dataset$causa , which(dataset$causa==3),2)
dataset$causa <- factor(dataset$causa,
                        labels=c("rayo", "negligencia", "intencionado",
                                 "causa desconocida","fuego reproducido"))
dataset$perdidas <- replace(dataset$perdidas ,
                            which(is.na(dataset$perdidas)),0)


dataset$causa_supuesta <- replace(dataset$causa_supuesta , which(is.na(dataset$causa_supuesta)),0)


## 3) LOCALIZACIÓN DE OUTLIERS - valores lógicos, ej, eliminar valores negativos en ciertas variables

# JUEVES REUNIÓN A LAS 18:00


# Meter primero exploración gráfica, conjunto de boxplot. Podemos hacer uno para las numériocas y otro para categóricas



#- Rango, máx, mínimo

## 4) Creación de variAbles


# Pendiente

# Buscar un gráfico que represente a la vez los boxplot para todas las variables, o uno para categoricas y otro numericas JORGE
# Dentro de los outliers ver los límites a partir de los que tenemos que eliminar  jORGE
# Ver la forma de crear un subset solo de numericas y otro de categóricas  ANA
# Crear columnas nuevas a partir de la de fecha de : trimestre, año, mes 
# En la causa detallada intentar sintetizar y buscar menos de 10 categorías ALEJANDRA
# Buscar los códigos de colinealidad ANA 
# Transformar las 2 de time a horas ALEJANDRA
# Tramificar personal y gastos ¿?
# Ver si se pueden sacar 4 categorías de la dirección del viento  ALEX
# Cambiar formato latitud y longitud

# COSAS PENDIENTES


# Transformar las 2 de time a horas ALEJANDRA

# Primero vemos las primeras filas de esta variable y el formato inicial

head(datos_galicia$time_ctrl)


# Creamos una variable puente para trabajar con ella y verificar que nos salen bien las transformaciones

totalMinutes = datos_galicia$time_ctrl

hour <- floor(totalMinutes / 60) # Redondea a la baja la hora
minute <- totalMinutes %% 60 * 0.01 # %% nos devuelve el resto de dividir entre 60
new_time <- hour + minute

datos_galicia$time_ctrl = new_time # Reemplazamos la variable que teníamos por la transformada

sort(datos_galicia$time_ctrl) # Comprobamos que al ordenar de forma creciente sí que funcióna!


# Repetimos el mismo proceso con time_ext 


totalMinutes2 = datos_galicia$time_ext

hour2 <- floor(totalMinutes2 / 60) 
minute2 <- totalMinutes2 %% 60 * 0.01 
new_time2 <- hour + minute

datos_galicia$time_ext = new_time2 

head(datos_galicia$time_ext)
sort(datos_galicia$time_ext)

str(datos_galicia)



## CREAR NUEVAS COLUMNAS EN FECHA CON TRIMESTRES, AÑOS Y MESES 

datos_galicia <- read.csv("Galicia.csv")

# Observamos que en la variable de fecha no hay NAs.por lo que no hay nada que omitir

# Lo convertimos en formato fecha para poder extraer los meses más adelante

datos_galicia$fecha = as.Date(datos_galicia$fecha, format ="%d/%m/%Y")

por_trimestres = datos_galicia %>% mutate(Quarter = quarters(fecha))
por_meses = por_trimestres %>% mutate(Month = months(fecha))
por_año = por_meses %>% mutate(Year = year(fecha))

datos_galicia = por_año



# Tramificar personal y gastos ¿?


str(datos_galicia)

min(datos_galicia$personal)
max(datos_galicia$personal)

# 5. Categorizar una variable cuantitativa en una cualitativa

boxplot(datos_galicia$personal)
boxplot(datos_galicia$gastos)
datos_galicia$personal<-replace(datos_galicia$personal, which((datos_galicia$personal < 0)|(datos_galicia$personal>400)), NA)



datos_galicia[,"personal_tramificado"] <- cut(datos_galicia$personal,breaks = c(25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400),
                                              labels = c("A","B","C","D","E","F","G","H","I","J","K","L","M","Ñ","O") ) 


