###############################################################################
############################# MODELO DE ARBOLES ###############################
###############################################################################



library(readr)
library(pacman)
p_load(tidyverse, readr, skimr, fastDummies, caret, glmnet, MLmetrics)


####DIVIDIMOS LA MUESTRA EN TRAIN Y TEST

set.seed(1022)
inTrain <- createDataPartition(
  y = train$Pobre,## La variable dependiente u objetivo 
  p = .7, ## Usamos 70%  de los datos en el conjunto de entrenamiento 
  list = FALSE)

train_muestra <- train[ inTrain,]
test_muestra  <- test[-inTrain,]


##ARBOLES---------------------------------------------------------------------

set.seed(123)


#Cross validation V=5

cv10<- trainControl(number = 10, method ="cv")

arbol_1<-train(Pobre ~ P6020+P6040+P6090+P6210+P6210s1+P6426+P6585s1+P6585s2+
                 P6585s3+P6800+P6920+Oc+P5000+P5010,
               data=train,
               method="rpart",
               trControl = cv10)

p_load(rattle)
fancyRpartPlot(arbol_1$finalModel)

pred_train_arbol_muestra<-predict(arbol_1, newdata=train_muestra)
pred_test_arbol_muestra<-predict(arbol_1, newdata=test_muestra)

write.csv(pred_train_arbol_muestra, "Arboles_Train.csv")
write.csv(pred_test_arbol_muestra, "Arboles_Test.csv")

#Error en porcentaje
MAPE(y_pred=pred_train_arbol_muestra, y_true = train_muestra$Pobre)
#0.3619813

#Error promedio
MAE(y_pred=pred_train_arbol_muestra, y_true = train_muestra$Pobre)
#213703743


## FUERA DE MUESTRA

#Error en porcentaje
MAPE(y_pred=pred_test_arbol_muestra, y_true = test_muestra$pobre)
#0.3605517

#Error promedio
MAE(y_pred=pred_test_arbol_muestra, y_true = test_muestra$pobre)
#213489943


# Se utilizaron modelos de arboles, dado que, estos aprenden las formas funcionales
# no lineales. A partir de particiones recursivas binarias, dividen el espacio variable
# por variable

# De los resultados, se evidencia que tras probar "hoja por hoja" se escoge el error
# estandar menor.

# Se busca el menor MAE.

# El MAE del modelo fue 213.489.943



##RANDOM FOREST------------------------------------------------------------

set.seed(123)

summary(train_muestra)

#Cross validation V=8

cv10<- trainControl(number = 10, method ="cv")

tunegrid_rf<-expand.grid(mtry=c(2,3,4,5, 8), #Predictores aleatorios
                         splitrule= "variance", ##Cambiar por gini y revisar
                         min.node.size=c(1,2,3,6))


rforest<-train(Pobre ~ P6020+P6040+P6090+P6210+P6210s1+P6426+P6585s1+P6585s2+
                 P6585s3+P6800+P6920+Oc+P5000+P5010,
               data=train_muestra, 
               trControl = cv10,
               metric = "RMSE",
               tuneGrid = tunegrid_rf,
               method ="ranger")


plot(rforest)

pred_train_2_muestra<-predict(rforest, newdata=train_muestra)
pred_test_2_muestra<-predict(rforest, newdata=test_muestra)

##EN MUESTRA
#Error en porcentaje
MAPE(y_pred=pred_train_2_muestra, y_true = train_muestra$Pobre)

#Error promedio
MAE(y_pred=pred_train_2_muestra, y_true = train_muestra$Pobre)

## FUERA DE MUESTRA
#Error en porcentaje
MAPE(y_pred=pred_test_2_muestra, y_true = test_muestra$pobre)

#Error promedio
MAE(y_pred=pred_test_2_muestra, y_true = test_muestra$pobre)

write.csv(pred_train_2_muestra, "Bosques_Train.csv")
write.csv(pred_test_2_muestra, "Bosques_Test.csv")
