## Load the dataset

url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data <- read.csv(url, header=FALSE)
head(data)

## add column names
colnames(data) <- c( "age","sex", "cp", "trestbps", "chol","fbs", "restecg", "thalach", 
                     "exang","oldpeak", "slope", "ca",  "thal","hd")

str(data)

## check missing
table(data == "?")
data[data == "?"] <- NA  # assign to missing
sapply(data, function(x) sum(is.na(x)))  # number of missing in the variables

## add factors for factor variables
data[data$sex == 0,]$sex <- "F"
data[data$sex == 1,]$sex <- "M"
data$sex <- as.factor(data$sex)

data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)

## 
data$ca <- as.integer(data$ca)
data$ca <- as.factor(data$ca)

data$thal <- as.integer(data$thal) 
data$thal <- as.factor(data$thal)

## outcome
data$hd <- ifelse(test=data$hd == 0, yes="Healthy", no="Unhealthy")
data$hd <- as.factor(data$hd)

## Build a decision tree
### impute missing values
library(randomForest)
set.seed(42)
data.imputed <- rfImpute(hd ~ ., data = data, iter=6)

## fit classification tree
library(tree)
set.seed(42)
tree<- tree(hd ~ ., data.imputed)
summary(tree) # only 10 variables used for the tree because of the stopping criteria. e.g. error rate set at certain cutoff
# 10% missclassification rate.

## present the results
par(mai=c(0,0,0,0))
plot(tree)
text(tree , pretty = 0)
 # most important predictors: that comes on the top of the tree are the most important
 # left side of branches/tree, less than or no; and the reverse for the right side

## training and test data
set.seed(42)
sample = sample (1: nrow(data.imputed), nrow(data.imputed)*0.8)
train.data= (data.imputed[sample,])
test.data =data.imputed[-sample ,] 
tree.train <- tree(hd ~ ., train.data)
tree.prediction <- predict(tree.train , test.data , type = "class")
Observed= test.data[,1]
table(tree.prediction, Observed)

## Tree pruning
set.seed(42)
cv <- cv.tree(tree.train , FUN = prune.misclass)
cv

## plot error rate
par(mfrow = c(1, 1))
plot(cv$size, cv$dev, type = "b", xlab="Tree size(model complexity)", ylab="Error rate")

## apply prone.misclass for the nine-node tree
set.seed(42)
prune.tree <- prune.misclass(tree.train , best = 10)
plot(prune.tree)
text(prune.tree , pretty = 0)

## again predict 
set.seed(42)
prune <- predict(prune.tree , test.data , type = "class")
Observed= test.data[,1]
table(prune, Observed)  

## Buikd random forest
set.seed(42)
model <- randomForest(hd ~ ., data=data.imputed, proximity=TRUE)
model 

## Gini Impurity Index
model$importance. # predictors with high mean decrease are best
varImpPlot(model)  # plot gini impurity
# it has more trees (500) than the decision tree that has only 1; better for predictor selection

## Tunning hyperparameters 
oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=3),
  Type=rep(c("OOB", "Healthy", "Unhealthy"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"], 
          model$err.rate[,"Healthy"], 
          model$err.rate[,"Unhealthy"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

## add more trees
set.seed(7)
model <- randomForest(hd ~ ., data=data.imputed, ntree=1000, proximity=TRUE)
model. # adding trees couldn't increase accuracy of the model

##
oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=3),
  Type=rep(c("OOB", "Healthy", "Unhealthy"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"], 
          model$err.rate[,"Healthy"], 
          model$err.rate[,"Unhealthy"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

## After building a random forest with 1,000 trees, we get the same OOB-error 16.5% and we can see convergence in the graph. If we want to compare this random forest to others with different values for mtry (to control how many variables are considered at each step), then:
oob.values <- vector(length=10)
for(i in 1:10) {
  temp.model <- randomForest(hd ~ ., data=data.imputed, mtry=i, ntree=1000)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values
min(oob.values)  # minimum error
which(oob.values== min(oob.values)). #  optimal value 

## create prediction model using the best value for mtry
model <- randomForest(hd ~ ., data=data.imputed,
                      ntree=1000, 
                      proximity=TRUE, 
                      mtry=3)
model

## Create multidimensional scaling (MDS) plot
set.seed(42)
 # Start by converting the proximity matrix into a distance matrix.
distance.matrix <- as.dist(1-model$proximity)
mds.stuff <- cmdscale(distance.matrix, eig=TRUE, x.ret=TRUE)

 # calculate the percentage of variation that each MDS axis accounts for.
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)

 # now make a fancy looking plot that shows the MDS axes and the variation.
set.seed(42)
# Start by converting the proximity matrix into a distance matrix.
distance.matrix <- as.dist(1-model$proximity)
mds.stuff <- cmdscale(distance.matrix, eig=TRUE, x.ret=TRUE)

 # calculate the percentage of variation that each MDS axis accounts for.
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)

 # now make a fancy looking plot that shows the MDS axes and the variation.
mds.values <- mds.stuff$points
mds.data <- data.frame(Sample=rownames(mds.values),
                       X=mds.values[,1],
                       Y=mds.values[,2],
                       Status=data.imputed$hd)

ggplot(data=mds.data, aes(x=X, y=Y, label=Sample)) + 
  geom_text(aes(color=Status)) +
  theme_bw() +
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep="")) +
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep="")) +
  ggtitle("MDS plot") +
  theme(plot.title = element_text(hjust = 0.5))
