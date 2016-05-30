
concrete <- read.csv("concrete.csv")
data1 <- concrete



#step1 - data split.  training set and testing set. 
set.seed(2)
inTraining <- createDataPartition(data1$X, p = .75, list = FALSE)
training <- data1[ inTraining,]
testing  <- data1[-inTraining,]

#step2 
summary(training)
training$X <- NULL
testing$X <- NULL
colnames(training) <- c("a", "b", "c", "d", "e", "f", "g", "h", "output")
colnames(testing) <- c("a", "b", "c", "d", "e", "f", "g", "h", "output")

#Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
#Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
#Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
#Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
#Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
#Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
#Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
#Age -- quantitative -- Day (1~365) -- Input Variable
#Concrete compressive strength -- quantitative -- MPa -- Output Variable 
# install.packages("ISLR")


library(AppliedPredictiveModeling)

transparentTheme(trans = .4)

featurePlot(x = training[, 1:8],
            y = training$output,
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))


ggplot(aes(x = a, y = output), data = training) + geom_jitter(alpha = 1, color = 'black') 
ggplot(aes(x = b, y = output), data = training) + geom_jitter(alpha = 1, color = 'black') 
ggplot(aes(x = c, y = output), data = training) + geom_jitter(alpha = 1, color = 'black') 
ggplot(aes(x = d, y = output), data = training) + geom_jitter(alpha = 1, color = 'black') 
ggplot(aes(x = e, y = output), data = training) + geom_jitter(alpha = 1, color = 'black') 
ggplot(aes(x = f, y = output), data = training) + geom_jitter(alpha = 1, color = 'black') 
ggplot(aes(x = g, y = output), data = training) + geom_jitter(alpha = 1, color = 'black') 
ggplot(aes(x = h, y = output), data = training) + geom_jitter(alpha = 1, color = 'black') 

library(caret)



#linear regression
#Penalized Linear Regression-'glmnet '
#k-Nearest Neighbors-"knn"
#CART - 'rpart'
#Support Vector Machines with Linear Kernel (method = 'svmLinear')
set.seed(2)
formula <- output ~ a + b + c + d + e + f + g + h
fit.lm <- train(formula, data=training ,method='lm')
fit.glm <- train(formula, data=training ,method='glmnet')
fit.knn <- train(formula, data=training ,method='knn')
fit.rpart <- train(formula, data=training ,method='rpart')
fit.svmLinear <- train(formula, data=training ,method='svmLinear')

plot(varImp(fit.lm))
plot(varImp(fit.glm))
plot(varImp(fit.knn))
plot(varImp(fit.rpart))
plot(varImp(fit.svmLinear))

resamps <- resamples(list(LM = fit.lm,
                          GLM = fit.glm,
                          KNN = fit.knn, 
                          RPART = fit.rpart,
                          SVM = fit.svmLinear))

summary(resamps)


 
bwplot(resamps)


#anaylsis training set.  cv 
set.seed(2)
ctrl<-trainControl(method = 'cv' ,number = 5)
fit.cv.lm<-train(output ~ ., data = training, method = 'lm' , trControl = ctrl, metric= 'RMSE')
fit.cv.glm <-train(output ~ ., data = training, method = 'glmnet' , trControl = ctrl, metric= 'RMSE')
fit.cv.knn <-train(output ~ ., data = training, method = 'knn' , trControl = ctrl, metric= 'RMSE')
fit.cv.rpart <-train(output ~ ., data = training, method = 'rpart' , trControl = ctrl, metric= 'RMSE')
fit.cv.svmLinear <-train(output ~ ., data = training, method = 'svmLinear' , trControl = ctrl, metric= 'RMSE')




plot(varImp(fit.cv.lm))
plot(varImp(fit.cv.glm))
plot(varImp(fit.cv.knn))
plot(varImp(fit.cv.rpart))
plot(varImp(fit.cv.svmLinear))

resamps.cv <- resamples(list(LM = fit.cv.lm,
                          GLM = fit.cv.glm,
                          KNN = fit.cv.knn, 
                          RPART = fit.cv.rpart,
                          SVM = fit.cv.svmLinear))
summary(resamps.cv)
bwplot(resamps.cv)

 

#tuning
set.seed(2)
knnGrid <- expand.grid(.k=c(2:10))
fit.knn.tune <- train(formula, data=training ,method='knn', tuneLength = 10, tuneGrid = knnGrid )
fit.knn.tune
plot(fit.knn.tune)



set.seed(2) 
knnGrid.final <- expand.grid(.k= 5)
fit.knn.final <- train(formula, data=training ,method='knn', tuneLength = 10, tuneGrid = knnGrid.final )
fit.knn.final 

#test with final model
set.seed(2)
testpredict <- predict(fit.knn.final, testing)
summary(testpredict)
plot(testing$output , testpredict)
postResample(testpredict, testing$output)




#preprocessing
set.seed(2)
preProc <- c("center", "scale")
fit.knn2 <- train(output ~ ., data = training, method = 'knn' , trControl = ctrl, metric= 'RMSE', preProc=preProc)
fit.knn2

set.seed(2)
fit.cv.knn2 <-train(output ~ ., data = training, method = 'knn' , trControl = ctrl, metric= 'RMSE', preProc = preProc)
fit.cv.knn2


set.seed(2)
knnGrid <- expand.grid(.k=c(2:10))
fit.knn2.tune <- train(formula, data=training ,method='knn', tuneLength = 10, tuneGrid = knnGrid, preProc=preProc  )
fit.knn2.tune
plot(fit.knn2.tune)

set.seed(2)
knnGrid.final <- expand.grid(.k= 5)
fit.knn2.final <- train(formula, data=training ,method='knn', tuneLength = 10, tuneGrid = knnGrid.final, preProc = preProc )
fit.knn2.final 


#test with final model
set.seed(2)
train.testpredict <- predict(fit.knn2.final, testing)
summary(train.testpredict)
plot(testing$output , train.testpredict)
postResample(train.testpredict, testing$output)


