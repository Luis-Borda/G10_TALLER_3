################################################################################
###################### MODELOS DE REGRESIÓN DE INGRESO #########################
################################################################################

####################### Modelos de Regularización ##############################
library(pacman)
library(leaflet)
library(stringi)
library(readr)

# Cargar las librerías listadas e instalarlas en caso de ser necesario
p_load(tidyverse, # Manipular dataframes
       rio, # Import data easily
       leaflet, # Mapas interactivos
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

#Importar datos

test <- read_csv("C:\Users\Hp\Documents\MeCA - Big Data and Machine Learning\Set_3\test_filtrada")
train <- read_csv("C:\Users\Hp\Documents\MeCA - Big Data and Machine Learning\Set_3\train_filtrada")

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

set.seed(123)  # Semilla para reproducibilidad

split <- initial_split(train, prop = 0.80) #Se divide la prueba para obtener un
# resultado provisional sobre la base de entrenamiento 

train_data <- training(split)
test_data <- testing(split)



fitControl <- trainControl(
  method = "cv",
  number = 10)

fmla<-formula(Ingtob ~ Sexo+Edad+Seguridads+Maxniveleduc+Tiempotrabajo+
                subaliment+auxtransp+subfam+
                hrstrabajadas+cotizafondopen)

linear_reg<-train(fmla,
                  data=train,
                  method = 'lm', 
                  trControl = fitControl
) 


linear_reg
summary(linear_reg)

y_hat_reg <- predict(linear_reg, newdata = test_data)

#------ Modelo Ridge --------------------------#

ridge<-train(fmla,
             data=train,
             method = 'glmnet', 
             trControl = fitControl,
             tuneGrid = expand.grid(alpha = 0, #Ridge
                                    lambda = seq(10000000, 20000000,by = 10000)),
             preProcess = c("center", "scale")
) 

plot(ridge$results$lambda,
     ridge$results$RMSE,
     xlab="lambda",
     ylab="Root Mean-Squared Error (RMSE)"
)

ridge$bestTune

coef_ridge<-coef(ridge$finalModel, ridge$bestTune$lambda)
coef_ridge

modelo_ridge<-train(fmla,
                    data=train,
                    method = 'glmnet', 
                    trControl = fitControl,
                    tuneGrid = expand.grid(alpha = 0, #Ridge
                                           lambda = 14880000),
                    preProcess = c("center", "scale")
) 

y_hat_ridge <- predict(modelo_ridge, newdata = test)

## Modelo Lasso

lasso<-train(fmla,
             data=train,
             method = 'glmnet', 
             trControl = fitControl,
             tuneGrid = expand.grid(alpha = 1, #lasso
                                    lambda = seq(10000,1000000,by = 1000)),
             preProcess = c("center", "scale")
) 

plot(lasso$results$lambda,
     lasso$results$RMSE,
     xlab="lambda",
     ylab="Root Mean-Squared Error (RMSE) Lasso"
)

lasso$bestTune

coef_lasso<-coef(lasso$finalModel, lasso$bestTune$lambda)
coef_lasso

modelo_lasso<-train(fmla,
                    data=train,
                    method = 'glmnet', 
                    trControl = fitControl,
                    tuneGrid = expand.grid(alpha = 1, #lasso
                                           lambda = 320000),
                    preProcess = c("center", "scale")
) 

y_hat_lasso <- predict(modelo_lasso, newdata = test)

## Elastic Net

EN<-train(fmla,
          data=train,
          method = 'glmnet', 
          trControl = fitControl,
          tuneGrid = expand.grid(alpha = seq(0,1,by = 0.1), #grilla de alpha
                                 lambda = seq(100000,10000000,by = 10000)),
          preProcess = c("center", "scale")
) 

EN$bestTune

coef_EN<-coef(EN$finalModel,EN$bestTune$lambda)
coef_EN

modelo_EN<-train(fmla,
                 data=train,
                 method = 'glmnet', 
                 trControl = fitControl,
                 tuneGrid = expand.grid(alpha = 0.9, #grilla de alpha
                                        lambda = 320000),
                 preProcess = c("center", "scale")
) 

y_hat_EN <- predict(modelo_EN, newdata = test)

## Tabla: Coeficientes de los modelos

coefs_df<-cbind(coef(linear_reg$finalModel),as.matrix(coef_ridge),as.matrix(coef_lasso),as.matrix(coef_EN))
colnames(coefs_df)<-c("OLS","RIDGE","LASSO","ELASTIC_NET")
round(coefs_df,4)

RMSE_df<-cbind(linear_reg$results$RMSE,ridge$results$RMSE[which.min(ridge$results$lambda)],lasso$results$RMSE[which.min(lasso$results$lambda)],EN$results$RMSE[which.min(EN$results$lambda)])
colnames(RMSE_df)<-c("OLS","RIDGE","LASSO","EN")
RMSE_df

# Para enviar valores de prediccion

# y_hat_reg
# y_hat_ridge
# y_hat_lasso
# y_hat_EN

price <- y_hat_lasso
submission_template <- select(submission_template, -price)
submission_template <- cbind(submission_template, price)
names(submission_template)[ncol(submission_template)] <- "price"

p4 <- ggplot(submission_template, aes(x = price)) +
  geom_histogram(bins = 50, fill = "darkblue", alpha = 0.4) +
  labs(x = "precio en pesos ($)", y = "Cantidad",
       title = "Distribución del precio") +
  theme_bw()
p4
