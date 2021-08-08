# Trabajo Fin de Máster

# DATASET INCENDIOS COMBINADO CON GALICIA

# ESTRUCTURA:

# 1. CREACIÓN DE DIRECTORIO DE TRABAJO E IMPORTACIÓN DE LOS DATOS

# 2. IMPORTACIÓN DE LIBRERÍAS

# 3. DESCRIPCIÓN GENERAL DEL DATASET : INSPECCIÓN ANALÍTICA Y VISUAL

# 4. ESTUDIO Y TRATAMIENTO DE VALORES PERDIDOS

# 5. ESTUDIO Y TRATAMIENTO DE OUTLIERS

# 6. TRANSFORMACIÓN DE VARIABLES

# 7. CREACIÓN DE VARIABLES



# 1. CREACIÓN DE DIRECTORIO DE TRABAJO E IMPORTACIÓN DE LOS DATOS


# Tenemos dos alternativas para la lectura de los datos: fijando un directorio de trabajo o leyendo los dataset desde una ubicación web

# Fijamos un directorio de trabajo:

setwd("C:/Users/Alicia López Machado/Desktop/Máster Big Data & Data Science/TFM")

# Leemos los dos dataset

datos <- read.csv("DATASET TFM_v2_12072021.csv")

datos_galicia <- read.csv("Galicia_definitivo.csv")



######### (Lena) Leer el dataset directamente de GitHub -- porfi podéis poner aquí el código que funcionaba?

library (readr) # librería para leer datos
url = "https://raw.githubusercontent.com/TFM123456/Big_Data_and_Data_Science_UCM/main/Galicia_definitivo.csv?token=APWX757FFLSZSWKCBLBC6SLBBAUWQ"
url2 = "https://github.com/TFM123456/Big_Data_and_Data_Science_UCM/blob/main/Galicia_definitivo.csv"
datos_galicia = as.data.frame(read.csv(url(url)))




# 2. IMPORTACIÓN DE LIBRERÍAS

# Vamos a importar las librerías que utilizaremos para la limpieza y el tratamiento de los datos y añadimos una nota breve para utilidad / finalidad de cada una.

# PONERLAS A MEDIDA QUE NOS LAS VAN PIDIENDO LAS FUNCIONES...



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
library(readr) # librería para leer datos


#library(leaflet)
#install.packages("leaflet")
#install.packages("ggplot2")
#install.packages("sf")
#library(sf)





## 1) DESCRIPCIÓN GENERAL DEL DATASET


# En este script tenemos contamos con 2 datasets. El primero de ellos, el de "datos" es aplicable al conjunto del territorio español, ya que las variables
#recogen información para todas las comunidades autónomas. Sobre este dataset,tenemos que decir que es compuesto, es decir, que se ha generado a partir de unos
# ya existente : xxxxxxxxxxxxxxxxxxxx al que hemos unido otro con información sobre variables mayoritariamente geográficas, para poder en un futuro, crear modelos
# más realistas y tener predicciones de mayor calidad, o con un r cuadrado mayor, entre otras razones.

# El dataset en el que se va a fundamentar el trabajo fin de máster es el de datos_galicia, que es un subset del dataset anterior. Hemos optado por trabajar con 
# la información recogida para Galicia, ya que es la Comunidad Autónoma en la que hemos comprobado que la disponibilidad de la información era mayor: ya sea por
# la mayor recogida de datos o la mayor incidencia del problema de incendios forestales.

# El preprocesado va a partir de el dataset_galicia, que se irá viendo modificado.



#1.1) Exploración inicial y tipo de datos / variables y transformación 

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

# Después de realizar las primeras transformaciones a los tipos de datos sin alterar su contenido, hacemos un análisis rápido del dataset y comprobar
# la no duplicidad de ids.

length(unique(datos_galicia$id)) # No tenemos valores duplicados

summary(datos_galicia)  # Vemos que aproximadamente la mitad de las variables tienen valores perdidos, que analizaremos luego.

#head(datos_galicia)
#dim(datos_galicia) 
#tail(datos_galicia)
glimpse(datos_galicia)
#psych::describe(Filter(is.numeric, datos_galicia))  usar para hacer un subset después de transformar

# Revisamos de nuevo las numéricas para ver si alguna más podría ser transformada en factor, pero en principio las mantenemos como están.

head(Filter(is.numeric, datos_galicia)) 

#1.2) Exploración gráfica inicial 


# Esta exploración la queremos realizar mediante la representación de un conjunto de diagramas de caja y bigotes:

# Gráficos de inspección rápida de un dataset. Opción para obtener boxplots

# PENDIENTE: Hacer el boxplot general con todas la variables

# Gráficos de inspección rápida de un dataset. Opción para obtener boxplots
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

# Lo podemos hacer seleccionando las columnas que queremos
listaGraf <- dfplot_box(datos_galicia[,c(-8)]) #Boxplots # Queremos eliminar la columna de los municipios

# Ver si nos interesa la representación del histograma

gridExtra::marrangeGrob(listaGraf, nrow = 4, ncol = 5)


listaHist<-dfplot_his(datos_galicia) #Histogramas
# pacman::p_load()

gridExtra::marrangeGrob(listaHist, nrow = 3, ncol = 2)




#¡ GUÍA


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


# altitud , nas outliers muchas por debajo del nivel del mar¿?
# t media outliers y na , mandar las del 99 a na , imputar por ?
# dirección del viento , tramificar¿?- en la parte de la creación de nuevas varibles, crear 4 categorías





listaGraf <- dfplot_box(datos_galicia) #Boxplots
listaGraf <- dfplot_box(datos_galicia[,c(1:4)]) #Boxplots
######################################################################

warning = FALSE




# 4. ESTUDIO Y TRATAMIENTO DE VALORES PERDIDOS
#----------------------------------------------#

# -- Localización -- #


total_perdidos = sapply(datos_galicia, function(datos_galicia) sum(is.na(datos_galicia))) # Cantidad de perdidos por variable

porcentaje_perdidos = 100*sapply(datos_galicia, function(datos_galicia) mean(is.na(datos_galicia))) # % de perdidos por variable


total_perdidos
porcentaje_perdidos


# Las variables de latitud, longitud y las variables geográficas/ geológicas presentan un porcentaje muy bajo de valores perdidos
# por lo que el tratamiento más directo será eliminar las filas en las que no tengan información para esos registros.


# Por contra, el porcentaje de valores perdidos de las variables de muertos, heridos, gastos y pérdidas es mucho más significativo
# por lo que tendremos que dar un tratamiento diferente en este caso.

# -- Tratamiento de NA -- #

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

# Tenemos dos opciones: eliminar las obervaciones o reemplazar por la media. Vamos a comprobar la dispersión de la media de cada
# una de estas dos columnas para ver si sería una acción adecuada reemplazar por este valor.

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


# 5. ESTUDIO Y TRATAMIENTO DE OUTLIERS     -   Acabar
#--------------------------------------#

# En este apartados queremos revisar el contenido delas variables y la posible presencia de valores extremos.
# Para ello haremos una exploración gráfica para cada una de las variables, que no sean dicotómicas y para las dicotómicas
# estudiaremos la presencia de valores distintos a los dos posibles que pueda adoptar la variable.


# Al igual que en el tratamiento de los valores perdidos, podremos tomar dos acciones: la eliminación de los valores extremos o el 
# reemplazo por la media o la media. Valoraremos en función de cada caso.

# En primer lugar haremos un subset con la variables numéricas y otro con las categóricas.
# Podemos ver los valores máximos y mínimos de cada una de las variables


numlab_df <- datos_galicia %>% select(is.numeric)
names(numlab_df)

# De este subset vamos a eliminar las que no deberían ser analizadas en cuanto a sus valores extremos, que serían:

# Id, idcomunidad, idprovincia, idmunicipio, causa, causa_desc, muertos, heridos. Eliminamos estas columnas del subset:

numlab_df <- numlab_df %>% select("superficie","lat","lng","time_ctrl","time_ext","personal","medios","perdidas",
                                  "ALTITUD","TMEDIA","PRECIPITACION","TMIN","TMAX","DIR","VELMEDIA","RACHA","SOL",
                                  "PRESMAX","PRESMIN")




# Representamos el boxplot de este dataframe con el código que teníamos al principio del script:


listaGraf <- dfplot_box(numlab_df) #Boxplots
gridExtra::marrangeGrob(listaGraf, nrow = 4, ncol = 5)

# En la mayoría de variables es evidente que hay valores extremos y a raíz del análisis gráfico, recibirán tratamientos distintos:

summary(numlab_df)

# Para la variable de superficie:

boxplot(datos_galicia$superficie)

length(numlab_df$superficie > 900)

datos_galicia$superficie<-replace(datos_galicia$superficie, which((datos_galicia$superficie < 0)|(datos_galicia$superficie>6)), NA)


# Para las variables de ALTITUD, TMAX, SOL, TMEDIA, DIR, PRESMAX, PRESMIN Y TMIN se puede ver gráficamente los outliers, así que vamos a eliminar esas observaciones:

datos_galicia$ALTITUD<-replace(datos_galicia$ALTITUD, which((datos_galicia$ALTITUD < 0)|(datos_galicia$ALTITUD>200)), NA)
datos_galicia$TMAX<-replace(datos_galicia$TMAX, which((datos_galicia$TMAX < 0)|(datos_galicia$TMAX>50)), NA)
datos_galicia$SOL<-replace(datos_galicia$SOL, which(datos_galicia$SOL<5), NA)
datos_galicia$TMEDIA<-replace(datos_galicia$TMEDIA, which(datos_galicia$TMEDIA<5), NA)
datos_galicia$DIR<-replace(datos_galicia$DIR, which((datos_galicia$DIR < 0)|(datos_galicia$DIR>36)), NA)
datos_galicia$PRESMAX<-replace(datos_galicia$PRESMAX, which(datos_galicia$PRESMAX < 250), NA)
datos_galicia$PRESMIN<-replace(datos_galicia$PRESMIN, which(datos_galicia$PRESMIN < 250), NA)
datos_galicia$TMIN<-replace(datos_galicia$TMIN, which((datos_galicia$TMIN < 0)|(datos_galicia$TMIN>50)), NA)                            






boxplot(datos_galicia$lat)








cat_df <- datos_galicia %>% select(is.character)
names(cat_df)

summary(datos_galicia)









# 6. TRANSFORMACIÓN DE VARIABLES
#---------------------------------#





#dataset$perdidas <- replace(dataset$perdidas ,
#                           which(is.na(dataset$perdidas)),0)




# Transformación de idprovincia, municipio y comunidad- categorizar y clasificar en provincias

datos_galicia$idprovincia[datos_galicia$idprovincia == 15] <- "A Coruña"
datos_galicia$idprovincia[datos_galicia$idprovincia == 27] <- "Lugo"
datos_galicia$idprovincia[datos_galicia$idprovincia == 32] <- "Ourense"
datos_galicia$idprovincia[datos_galicia$idprovincia == 36] <- "Pontevedra"


datos_galicia$idcomunidad[datos_galicia$idcomunidad == 3] <- "Galicia"


datos_galicia$idmunicipio <- datos_galicia$municipio # y eliminamos municipio

datos_galicia$municipio = NULL

str(datos_galicia)


# Transformación de causas a char 


datos_galicia$causa <- replace(datos_galicia$causa , which(datos_galicia$causa==3),2)
datos_galicia$causa <- factor(datos_galicia$causa,
                              labels=c("rayo", "negligencia", "intencionado",
                                       "causa desconocida","fuego reproducido"))


datos_galicia$causa = as.character(datos_galicia$causa)


table(datos_galicia$causa)

# dicotimizar causa supuesta

datos_galicia$causa_supuesta <- replace(datos_galicia$causa_supuesta , which(is.na(datos_galicia$causa_supuesta)),0)









## CREAR NUEVAS COLUMNAS EN FECHA CON TRIMESTRES, AÑOS Y MESES 

#datos_galicia <- read.csv("Galicia.csv")

# Observamos que en la variable de fecha no hay NAs.por lo que no hay nada que omitir

# Lo convertimos en formato fecha para poder extraer los meses más adelante

datos_galicia$fecha = as.Date(datos_galicia$fecha, format ="%d/%m/%Y")

por_trimestres = datos_galicia %>% mutate(Quarter = quarters(fecha))
por_meses = por_trimestres %>% mutate(Month = months(fecha))
por_año = por_meses %>% mutate(Year = year(fecha))

datos_galicia = por_año

head(datos_galicia)

# Separación de numéricas y categóricas



str(datos_galicia)

numlab_df <- datos_galicia %>% select(is.numeric)
names(numlab_df)

cat_df <- datos_galicia %>% select(is.character)
names(cat_df)




# En la causa detallada intentar sintetizar y buscar menos de 10 categorías ALEJANDRA



# Buscar los códigos de colinealidad ANA 

corr_num<-corrplot(cor(numlab_df), use="pairwise", method = "ellipse",type = "upper") # Meter después de los NAs




# Tranformación de las variables de tiempo

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


datos_galicia$time_ext

# Tramificar personal y gastos ¿?


str(datos_galicia)

min(datos_galicia$personal)
max(datos_galicia$personal)

# 5. Categorizar una variable cuantitativa en una cualitativa - no tiene sentido

boxplot(datos_galicia$personal)
boxplot(datos_galicia$gastos)
datos_galicia$personal<-replace(datos_galicia$personal, which((datos_galicia$personal < 0)|(datos_galicia$personal>400)), NA)



datos_galicia[,"personal_tramificado"] <- cut(datos_galicia$personal,breaks = c(25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400),
                                              labels = c("A","B","C","D","E","F","G","H","I","J","K","L","M","Ñ","O") ) 


# Ver si se pueden sacar 4 categorías de la dirección del viento  ALEX - investigar - quitar NA primero


datos_galicia$DIR<-replace(datos_galicia$DIR, which((datos_galicia$DIR < 0)|(datos_galicia$DIR>36)), 0)



datos_galicia[,"viento"] <- cut(datos_galicia$DIR,breaks = c(0,4.5,9,13.5,18,22.5,27,31.5,36),
                                labels = c("A","B","C","D","E","F","G","H") ) 
table(datos_galicia$viento)

                        
                        
#datos_galicia[,"viento"] <- cut(datos_galicia$DIR,breaks = c(0, 2.25, 6.75, 11.25, 15.75, 20.25, 24.75, 29.25, 33.75, 36),
                                #labels = c("N","NE","E","SE","S","SW","W","NW","N") ) 

#Se realizan rupturas de 4.5º por cada dirección.
#0,36 = N -> Por tanto el Norte comprende entre 33.25 y 2.25
#4,5 = NE -> (2.25 - 6.75)
#9 = E -> (6.75 - 11.25)
#13,5 = SE (11.25 - - 15.75)
#18 = S (15.75 - 20.25)
#22,5 = SW -> (20.25 - 24.75)
#27 = W -> (24.75 - 29.25)
#31,5 = NW -> (29.25 - 33.75)
                        
                        
                        
# V DE CRAMER

graficoVcramer<-function(matriz, target){
  salidaVcramer<-sapply(matriz,function(x) Vcramer(x,target))
  barplot(sort(salidaVcramer,decreasing =T),las=2,ylim=c(0,1))
}

graficoVcramer(input,varObjCont)
seed(123) 
input$aleatorio<-runif(nrow(input)) 
input$aleatorio2<-runif(nrow(input)) 
corrplot(cor(cbind(varObjCont,Filter(is.numeric, input)), use="pairwise", 
             method="pearson"), method = "ellipse",type = "upper")
