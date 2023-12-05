################################################################################
###################### MODELOS DE REGRESIÓN DE INGRESO #########################
################################################################################

############################### REDES NEURONALES ###############################

library(pacman)
library(leaflet)
library(stringi)
library(ggplot2)
library(readr)

# Cargar las librerías listadas e instalarlas en caso de ser necesario
p_load(tidyverse, # Manipular dataframes
       rio, # Import data easily
       plotly, # Gráficos interactivos
       leaflet, # Mapas interactivos
       rgeos, # Calcular centroides de un poligono
       tmaptools, # geocode_OSM()
       sf, # Leer/escribir/manipular datos espaciales
       osmdata, # Traer info geo espacial (limites bogta, etc) 
       tidymodels, #para modelos de ML
       skimr,
       stargazer, #Mostrar modelos de ML
       VIM,
       discrim,
       kknn,
       stargazer,
       yardstick,
       GGally,
       rattle,
       randomForest,
       C50,
       caret,
       naivebayes,
       keras,
       caret,
       ) 


########################### Se importan los datos #############################

# Importar datos

test <- read_csv("C:/Users/Hp/Documents/MeCA - Big Data and Machine Learning/Set_3/test_filtrada.csv")
train <- read_csv("C:/Users/Hp/Documents/MeCA - Big Data and Machine Learning/Set_3/train_filtrada.csv")

# Análisis de Datos

skim(test)
skim(train)

test %>%
  summarise_all(~sum(is.na(.)))
train %>%
  summarise_all(~sum(is.na(.)))

train<- train %>% mutate(Edad2=Edad^2)
train<- train %>% mutate(Sexo= case_when(Sexo==1 ~"Hombre",
                                   Sexo==2 ~"Mujer"),
                         subaliment= case_when(subaliment==1 ~"recibe",
                                               subaliment==2 ~"no recibe"),
                         auxtransp= case_when(auxtransp==1 ~"recibe",
                                              auxtransp==2 ~"no recibe"),
                         subfam= case_when(subfam==1 ~"recibe",
                                              subfam==2 ~"no recibe"),
                         cotizafondopen= case_when(cotizafondopen==1 ~"recibe",
                                           cotizafondopen==2 ~"no recibe")
                         )
                    
# Definir las variables categoricas

variables_categoricas <- c("Sexo",
                           "subaliment",
                           "auxtransp",
                           "subfam",
                           "cotizafondopen")

train <- train %>% mutate_at(variables_categoricas, as.factor)

# Dividimos nuestra muestra en entrenamiento y pruebas, luego seleccionamos 
# nuestros vectores x, y. En este caso trabajamos con vectores de datos que son 
# estructuras más eficientes que las redes pueden capturar. Aparte esta división 
# la realizamos para preprocesar cada elemento aparte.

# División en conjunto de entrenamiento y prueba

set.seed(123)  # Semilla para reproducibilidad

split <- initial_split(train, prop = 0.80) #Se divide la prueba para obtener un
# resultado provisional sobre la base de entrenamiento 

train_data <- training(split)
test_data <- testing(split)

# Separar las variables predictoras (regresores) y la variable objetivo

x_train <-train_data %>% select( -Ingtob)
y_train <-train_data %>% pull(Ingtob)
x_test <-test_data %>% select( -Ingtob)
y_test <-test_data %>% pull(Ingtob)


# Estandarización Z = (X-u)/o

rec <- recipe(~., data = x_train) %>% #Receta
  step_normalize(all_numeric())

# Aplicar el preprocesamiento para normalizar los datos
x_test<- prep(rec) %>% bake(new_data = x_test)

# Aplicar el preprocesamiento para normalizar los datos
x_train <- prep(rec) %>% bake(new_data = x_train)

head(x_train)

## convertir en doomys para matrices, las redes neuronales reciben matrices 
## de datos.

x_test <- x_test %>%
  model.matrix(~ . - 1, data = .)  # Convertir a variables dummy

# Convertir columnas categóricas a variables dummy
x_train<- x_train %>%
  model.matrix(~ . - 1, data = .)  # Convertir a variables dummy

# Lista de columnas categóricas (factores) a convertir a variables dummy

columnas_factor <- c("Sexo",
                     "subaliment",
                     "auxtransp",
                     "subfam",
                     "cotizafondopen")


#------------------------------------------------------------------------------#
####################### Modelo de red neuronal simple ##########################
#------------------------------------------------------------------------------#

model_simple <- keras_model_sequential() %>%
  layer_dense(units = 1, input_shape = ncol(x_train))

plot(model_simple, show_shapes = TRUE, show_layer_names = TRUE)

model_simple %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_rmsprop(),
  metrics = c("mean_squared_error")
)

# Al compilar el modelo, se define la función de pérdida y se selecciona un 
# optimizador para que el modelo esté listo para ser entrenado.

# Entrenamos el modelo

model_simple %>% fit(x_train, y_train, epochs = 5, verbose = 0,batch_size = 32)
predictions <- model_simple %>% predict(x_test)
rmse_result <- sqrt(mean((predictions - y_test)^2))
print(paste("RMSE:", rmse_result))

# En el contexto de modelos de redes neuronales, un RMSE más bajo es deseable, 
# ya que indica que las predicciones del modelo se desvían menos de los valores 
# reales. 

################################ MEJORAS EN LA RED ############################

# Jugar con la capa de salida, se juega con algunas funciones de activación:##

############################################################################

## ReLU (Rectified Linear Unit) f(x)=max(0,x)

# Es eficiente computacionalmente, su capacidad para mitigar el problema de 
# gradientes que desaparecen y su habilidad para introducir no linealidad en 
# las redes neuronales, permitiendo que aprendan representaciones más complejas
# de los datos.

###################### Modelo de red neuronal - ReLU ###########################

model_simple_relu <- keras_model_sequential() %>%
  layer_dense(units = 1, input_shape = ncol(x_train), activation = "relu")

# Graficar la arquitectura del modelo

plot(model_simple_relu, show_shapes = TRUE, show_layer_names = TRUE)

model_simple_relu

model_simple_relu %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_rmsprop(),
  metrics = c("mean_squared_error")
)

# Entrenamos el modelo

model_simple_relu %>% fit(x_train, y_train, epochs = 5, verbose = 0,
                          batch_size = 32)

predictions <- model_simple_relu %>% predict(x_test)

rmse_result <- sqrt(mean((predictions - y_test)^2))

print(paste("RMSE:", rmse_result))


######################## Modelo de red neuronal tanh ###########################

# Modelo de red neuronal tangencial

model_simple_tanh<- keras_model_sequential() %>%
  layer_dense(units = 1, input_shape = ncol(x_train), activation = "tanh")

model_simple_tanh %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_rmsprop(),
  metrics = c("mean_squared_error")
)

# Entrenamos el modelo

model_simple_tanh %>% fit(x_train, y_train, epochs = 5, verbose = 0,batch_size = 32)
predictions <- model_simple_tanh %>% predict(x_test)
rmse_result <- sqrt(mean((predictions - y_test)^2))
print(paste("RMSE:", rmse_result))


################### INCREMENTAR LA PROFUNDIDAD DEL MODELO ######################

# Definir el modelo de regresión

model_profundo <- keras_model_sequential() %>%
  layer_dense(units = 64, input_shape = ncol(x_train) , activation = "relu") %>%
  layer_dense(units = 1)  # Capa de salida para regresión

# No se especifica una función de activación aquí, lo que significa que por 
# defecto se utilizará una función lineal o identidad para la capa de salida, ya
# que es una tarea de regresión y se busca predecir un valor numérico directamente.

# Graficar la arquitectura del modelo

plot(model_profundo, show_shapes = TRUE, show_layer_names = TRUE)

model_profundo

# Compilar el modelo

model_profundo %>% compile(
  loss = "mean_squared_error",  # Función de pérdida para regresión
  optimizer = optimizer_rmsprop(),  # Selecciona el optimizador adecuado
  metrics = c("mean_squared_error")
)

# Entrenamos el modelo
model_profundo %>% fit(x_train, y_train, epochs = 5, verbose = 2)

## hacemos una predicction
predictions <- model_profundo  %>% predict(x_test)

## ver el resultado
rmse_result <- sqrt(mean((predictions - y_test)^2))
print(paste("RMSE:", rmse_result))

# Vemos una mejora clara en el RMSE. seguimos mejorando para crear un modelo 
# mas preciso.

########################## Modelo Avanzado #####################################

# Definir el modelo de regresión
model <- keras_model_sequential() %>%
  layer_dense(units = 64, input_shape = ncol(x_train) , activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)  # Capa de salida para regresión

# Graficar la arquitectura del modelo
plot(model, show_shapes = TRUE, show_layer_names = TRUE)

model

# Compilar el modelo
model %>% compile(
  loss = "mean_squared_error",  # Función de pérdida para regresión
  optimizer = optimizer_rmsprop(),  # Selecciona el optimizador adecuado
  metrics = c("mean_squared_error")
)

# Entrenamos el modelo

model %>% fit(x_train, y_train, epochs = 15, verbose = 2)
score <- model %>% evaluate(x_test, y_test, verbose = 0)
cat('Test loss:', score["loss"], "\n")

# Predicciones

predictions <- model %>% predict(x_test)
rmse_result <- sqrt(mean((predictions - y_test)^2))
print(paste("RMSE:", rmse_result))


