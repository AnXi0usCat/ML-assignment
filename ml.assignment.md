---
title: "Machine Learning Assignment"
author: "Mihhail Fjodorov"
date: "Sunday, July 26, 2015"
output: html_document
---

***Summary***
The purpose of this report is to describe the procedure of creating a Machine Learning Algorithm
for prediction. I explored several possibilites as Classification Trees and Random Forest.
I split the training data 50/50 to get the out of Sample Classification Error. I was able to achieve
Accuracy of around 89 % with Trees and Accuracy of around 99 % with Random Forests.
I used the Random Forests Algorithm on the Validation Data Set and got a result of 20/20.

***Assignment and Data***
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

***Getting the Data***


```r
# Load the required Packages
library(caret)
library(rpart)
library(ggplot2)
library(randomForest)
library(rattle)

# url for the training data set

url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
# url for the vaildation data set

url2 <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
# download both data sets

download.file(url, destfile = "./training.csv", quiet = F)
download.file(url2, destfile = "./validation.csv", quiet = F)
# read both data sets in to R

training <- read.csv("./training.csv", header = T,na.strings = c("","NA","#DIV/0!"))
validation <- read.csv("./validation.csv", header = T,na.strings = c("","NA","#DIV/0!"))
```

***Variable Transformations and Data Cleaning***

In order to identify the best predictors I have excluded all of the features which could be considered to have low variance. Several features had a lot of NA values so I dedided to exclude those features as well.


```r
zeroVar <- nearZeroVar(training,saveMetrics = T)
training <- training[,!zeroVar[,4]]
validation <- validation[,!zeroVar[,4]]
nas <- colSums(sapply(training, is.na))
training <- training[,nas == 0]
validation <- validation[,nas == 0]
```


Furthermore, I have excluded the X variable from the data set since it is just an index and has no predictive value. Additionaly, I decided to leave the timestamp variables and the username variables in because without them the accuracy fell atleast 40%. I am a bit concerned about leaving the username becuase the algorithm might
expreince issue with scaling on the new data set.
Finally I have converted all of the data to Numeric since Random Forests prefers all of the features to be of
the same type.


```r
training_2 <- training[,-c(1,6)]
validation_2 <- validation[,-c(1,6)]
for(i in 1:(dim(training_2)[2]-1)){training_2[,i] <- as.numeric(training_2[,i])}
for(i in 1:(dim(validation_2)[2]-1)){validation_2[,i] <- as.numeric(validation_2[,i])}
```

I split the Training Data into Training and Testing and decided to use the offical Testing data as a Validation
Set. I have used 50/50 split becuase I dont have enough RAM to work with a larger Data Set.


```r
inTrain <- createDataPartition(training_2$classe, p = 0.5, list = F) 
train <- training_2[inTrain,]
test <- training_2[-inTrain,]
```

***ML Alorithms***

Since the replationship between the outcome and predictors does not appear to be linear
I decided to use the Classification Trees.


```r
# use classification trees for the initial model
modFit_tree <- rpart(classe ~ ., data=train, method="class")
prediction_tree <- predict(modFit_tree, test, type = "class")
confusionMatrix(prediction_tree, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2502   58    3   19   17
##          B   41 1503  117   51  124
##          C   73  273 1484  254  108
##          D  154   55  103 1213   54
##          E   20    9    4   71 1500
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8361          
##                  95% CI : (0.8286, 0.8434)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7933          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8968   0.7919   0.8673   0.7544   0.8319
## Specificity            0.9862   0.9579   0.9126   0.9554   0.9870
## Pos Pred Value         0.9627   0.8186   0.6770   0.7682   0.9352
## Neg Pred Value         0.9601   0.9505   0.9702   0.9520   0.9631
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2550   0.1532   0.1513   0.1236   0.1529
## Detection Prevalence   0.2649   0.1872   0.2234   0.1610   0.1635
## Balanced Accuracy      0.9415   0.8749   0.8900   0.8549   0.9095
```

Creating a Tree Plot to visualize the spilt

```r
fancyRpartPlot(modFit_tree)
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-1.png) 

Classification Trees worked quite well, but I think we can do even better with Ranom Forests.





```r
## random forests
modFit_forest <- randomForest(classe ~. , data=train)
prediction_forest <- predict(modFit_forest, test, type = "class")
confusionMatrix(prediction_forest, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2788    0    0    0    0
##          B    2 1896    9    0    0
##          C    0    2 1702   13    0
##          D    0    0    0 1593    7
##          E    0    0    0    2 1796
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9964         
##                  95% CI : (0.995, 0.9975)
##     No Information Rate : 0.2844         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9955         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9989   0.9947   0.9907   0.9961
## Specificity            1.0000   0.9986   0.9981   0.9991   0.9998
## Pos Pred Value         1.0000   0.9942   0.9913   0.9956   0.9989
## Neg Pred Value         0.9997   0.9997   0.9989   0.9982   0.9991
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1933   0.1735   0.1624   0.1831
## Detection Prevalence   0.2842   0.1944   0.1750   0.1631   0.1833
## Balanced Accuracy      0.9996   0.9988   0.9964   0.9949   0.9979
```

Random Forests turn out to be a much better predictor and we will use it on the validation data set


```r
# predict the values on the validation data set
answers <- predict(modFit_forest, validation_2, type = "class")

# submission code
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}
pml_write_files(answers)
```
