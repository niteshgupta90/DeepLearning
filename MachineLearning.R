library(h2o)
h2o.init(nthreads = -1)
train_file<-"http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
train<-h2o.importFile(train_file)
test_file<-"http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"
test<-h2o.importFile(test_file)
summary(train)
summary(test)
y <- "C65"
x <- setdiff(names(train), y)
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])
splits <- h2o.splitFrame(train, 0.80, seed=12345)
model <- h2o.deeplearning(
         x = x, 
         y = y, 
         training_frame = splits[[1]],
         validation_frame = splits[[2]],
         loss = "Quadratic",   
         distribution = "multinomial",
         activation = "Tanh", 
         hidden = c(300,300,300),
         adaptive_rate = FALSE,
         rate = 0.01,
         momentum_start = 0.2,
         input_dropout_ratio = 0.2, 
         sparse = TRUE,
         l1 = 1e-5, 			#regularization
         epochs = 100)

model@parameters
h2o.confusionMatrix(model,train)
h2o.confusionMatrix(model,test)
prediction <- h2o.predict(model, newdata = test)
accuracy <- sum(test[,y] == prediction$predict)/nrow(test)
print(accuracy)
model@model$scoring_history

#To Print Class Accuracy
rownames = c("Class0", " Class1", " Class2"," Class3"," Class4"," Class5"," Class6"," Class7"," Class8"," Class9")
colnames = c("Accuracy")

CM<- h2o.confusionMatrix(model, train)
Error<-CM$Error
Accuracy<-1 - Error
result <- array(c(Accuracy),dim = c(1,10,1),dimnames = list(colnames,rownames))
print(result)

CM<- h2o.confusionMatrix(model, test)
Error<-CM$Error
Accuracy<-1 - Error
result <- array(c(Accuracy),dim = c(1,10,1),dimnames = list(colnames,rownames))
print(result)


model <- h2o.deeplearning(
         x = x,
         y = y,
         training_frame = splits[[1]],
         validation_frame = splits[[2]],
         loss = "CrossEntropy",
         distribution = "multinomial",
         activation = "Tanh",
         hidden = c(100,100,100),
         adaptive_rate = FALSE,
         rate = 0.01,
         momentum_start = 0.2,
         input_dropout_ratio = 0.2,
         sparse = TRUE,
         l1 = 1e-5, 			#regularization
         epochs = 100)

model@parameters
h2o.confusionMatrix(model,train)
h2o.confusionMatrix(model,test)
prediction <- h2o.predict(model, newdata = test)
accuracy <- sum(test[,y] == prediction$predict)/nrow(test)
print(accuracy)
model@model$scoring_history

#To Print Class Accuracy
rownames = c("Class0", " Class1", " Class2"," Class3"," Class4"," Class5"," Class6"," Class7"," Class8"," Class9")
colnames = c("Accuracy")

CM<- h2o.confusionMatrix(model, train)
Error<-CM$Error
Accuracy<-1 - Error
result <- array(c(Accuracy),dim = c(1,10,1),dimnames = list(colnames,rownames))
print(result)

CM<- h2o.confusionMatrix(model, test)
Error<-CM$Error
Accuracy<-1 - Error
result <- array(c(Accuracy),dim = c(1,10,1),dimnames = list(colnames,rownames))
print(result)
