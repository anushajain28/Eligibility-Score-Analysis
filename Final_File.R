install.packages("caTools")
install.packages("forecast")
install.packages("utility")
library(forecast)
library(caret)
library(ade4)
library(randomForest)
library(caTools)
library(e1071)
library(rpart)
library(ggplot2)
library(rattle)
library(readr)
library(xgboost)
library(utility)

#loading the train data
prudential_train_Data <- read.csv(file="C:/Northeastern University/Sem - 3/ADS_SD/Mid Term/train.csv", header=TRUE, sep= ',')

#counting number of nulls in each column
count_null_columns <- sapply(prudential_train_Data, function(x) sum(is.na(x)))
count_null_columns[count_null_columns > nrow(prudential_train_Data)*0.4]
#removing columns with more than 30% if null values
Filtered_Data <- prudential_train_Data[,-c(30,35,36,38,48,53,62,70)]


#removing outliers
model = lm(Response~. , data = Filtered_Data)
cooksd <- cooks.distance(model)
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")

influential <- as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))])  # influential row numbers
influencers <- Filtered_Data[influential, ] #influential observations
Filtered_Data_new<- Filtered_Data[-influential, ]



#replacing null values with mean
for(i in 1:ncol(Filtered_Data_new))
{
  if(is.numeric(Filtered_Data_new[,i])) {
    Filtered_Data_new[is.na(Filtered_Data_new[,i]),i] <- mean(Filtered_Data_new[!is.na(Filtered_Data_new[,i]),i])
  }
}

# Removing variable with near to Zero variance 
nzv <- nearZeroVar(Filtered_Data_new, saveMetrics = FALSE)
Filtered_Data_new <- Filtered_Data_new[,-nzv]


#variable selection
lm_one <- lm(Response ~ Ins_Age,data=Filtered_Data_new)
lm_all <- lm(Response ~ .,data=Filtered_Data_new)
step(lm_one,scope=list(upper=lm_all, lower= lm_one), direction = "both",trace = 1)

Filtered_new <- Filtered_Data_new[,c("Ins_Age" , "BMI" , "Medical_History_4" , "Medical_History_39" , 
                                     "Medical_History_23" , "Product_Info_4" , "Medical_History_13" , 
                                     "InsuredInfo_6" , "Medical_History_18" , 
                                     "Medical_Keyword_23" , "Family_Hist_4" , "Medical_History_28" , 
                                     "Medical_History_3" , "Employment_Info_3" , "Medical_History_1" , 
                                     "Insurance_History_1" , "Medical_Keyword_48" , "Medical_Keyword_37" , 
                                     "Medical_Keyword_25" , "Product_Info_3" , "InsuredInfo_1" , "Medical_History_21" , 
                                     "Medical_History_41" , "Medical_History_12" , "Product_Info_6" , 
                                     "Medical_History_37" , "Medical_History_2" , "Product_Info_2" , 
                                     "Family_Hist_1" , "Insurance_History_8","Response")]


Filtered_cat <-acm.disjonctif(Filtered_new[c(3,4,5,7,8,9,12,13,14,16,20,21,22,23,24,25,26,27,28,29,30)])
Filtered_new <- Filtered_new[,-c(3,4,5,7,8,9,12,13,14,16,20,21,22,23,24,25,26,27,28,29,30)]
Final_data <- cbind(Filtered_new,Filtered_cat)


set.seed(123)
index = sample(seq_len(nrow(Final_data)), size = floor(0.8*nrow(Final_data)))
train_data <- Final_data[index, ]
test_data <- Final_data[-index, ]
write.csv(Final_data, file = "C:/Northeastern University/Sem - 3/ADS_SD/Mid Term/Filtered_train_new_test.csv", row.names = FALSE)


#building Linear Model
linear_model = lm(Response~., data = train_data)
summary(linear_model)
plot(linear_model)

pred_response <- predict( linear_model, test_data)

for(i in 1:length(pred_response)) {
  pred_response[i] = floor(pred_response[i])
}

rmse <- sqrt(sum((pred_response-test_data$Response)^2)/length(pred_response))
rmse

accuracy(test_data$Response,pred_response)


table(pred_response,test_data$Response)

plot(pred_response,test_data$Response,
     xlab="predicted",ylab="actual")


# Decision Tree 
dec_tree <- rpart(Response ~ ., data = train_data, control = rpart.control(minsplit = 200,cp=0.01),method = "anova")
summary(dec_tree)
plot(dec_tree)
text(dec_tree, use.n=TRUE, all=TRUE, cex=.4)


fancyRpartPlot(dec_tree)
printcp(dec_tree)
plotcp(dec_tree)

prune_tree<- prune(dec_tree, cp = dec_tree$cptable[which.min(dec_tree$cptable[,"xerror"]),"CP"])
plot(prune_tree, uniform = TRUE, main = "PRUNED TREE")
text(prune_tree, use.n=TRUE, all=TRUE, cex=.4)
fancyRpartPlot(prune_tree,uniform = TRUE, main = "PRUNED TREE")
plotcp(prune_tree)
printcp(prune_tree)

prediction_tree <- predict(dec_tree, test_data)
for(i in 1:length(prediction_tree)) {
  prediction_tree[i] = round(prediction_tree[i])
}

rmse_tree <- sqrt(sum((prediction_tree-test_data$Response)^2)/length(prediction_tree))
rmse_tree
accuracy(prediction_tree,test_data$Response)
