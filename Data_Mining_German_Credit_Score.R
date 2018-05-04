#load the dataset
credit <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
View(credit)
#give column names
colnames(credit) = c("chk_acct", "duration", "credit_his", "purpose", 
                     "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                     "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                     "job", "n_people", "telephone", "foreign", "response")

# orginal response coding 1= good, 2 = bad we need 0 = good, 1 = bad
credit$response = credit$response - 1


#random sampling 
set.seed(12434842)
subset <- sample(nrow(credit), nrow(credit) * 0.75) #75% sampling
credit_train = credit[subset, ] #train
credit_test = credit[-subset, ] #test
#train- 750 obs, test- 250 obs


#cost function
cost <- function(observed, predicted) {
  weight1 = 5
  weight0 = 1
  c1 = (observed == 1) & (predicted == 0)  #logical vector - true if actual 1 but predict 0
  c0 = (observed == 0) & (predicted == 1)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}

#general linear model- logistic

credit_glm <- glm(response ~ ., family = binomial, credit_train)
summary(credit_glm)


#step wise variable selection
credit_glm_step <- step(credit_glm, direction = c("both"))

#revised model- stepwise output
credit_glm_final <- glm(response ~ chk_acct + duration + credit_his + purpose + amount + 
                          saving_acct + present_emp + installment_rate + other_install + 
                          housing + foreign, family = binomial, credit_train)

summary(credit_glm_final)

#in sample- prediction
in_samp_prob_glm <- predict(credit_glm_final, credit_train, type = "response")
in_samp_predict_glm <- in_samp_prob_glm > 1/6
in_samp_predict_glm <- as.numeric(in_samp_predict_glm)

#ROC
install.packages("ROCR")
library(ROCR)
#in sample
predinsamp <-prediction(in_samp_prob_glm, credit_train$response)
perfinsamp <- performance(predinsamp, "tpr","fpr")
plot(perfinsamp,colorize = TRUE)
as.numeric(performance(predinsamp,'auc')@y.values) #0.8169


#model deviance
credit_glm_final$deviance

#table
table(credit_train$response,in_samp_predict_glm, dnn = c("Truth","Predicted")) # 260, 254,21,215 - (0,0),(0,1),(1,0),(1,1)

#misclassification rate 
mean(ifelse(credit_train$response != in_samp_predict_glm,1,0)) #0.3573

#cost function
cost(credit_train$response,in_samp_predict_glm) #0.4693

#out of sample prediction
out_samp_prob_glm <- predict(credit_glm_final,newdata = credit_test, type = "response")
out_samp_predict_glm <- out_samp_prob_glm > 1/6
out_samp_predict_glm <- as.numeric(out_samp_predict_glm)

#ROC- out of sample
predoutsamp <-prediction(out_samp_prob_glm, credit_test$response)
perfoutsamp <- performance(predoutsamp, "tpr","fpr")
plot(perfoutsamp,colorize = TRUE)
as.numeric(performance(predoutsamp, 'auc')@y.values) #0.8211

#table
table(credit_test$response,out_samp_predict_glm, dnn = c("Truth","Predicted")) # 94,92,8,56 - (0,0),(0,1),(1,0),(1,1)

#misclassification rate 
mean(ifelse(credit_test$response != out_samp_predict_glm,1,0)) #0.1056

#cost function
cost(credit_test$response,out_samp_predict_glm) #0.528


#classification tree
install.packages("rpart")
library(rpart)
credit.rpart <- rpart(formula = response ~ ., data = credit_train, method = "class", 
                      parms = list(loss = matrix(c(0, 5, 1, 0), nrow = 2)))

#plot tree
plot(credit.rpart)
text(credit.rpart)

plotcp(credit.rpart) #cp =0.013, nodes =11
#revised tree
credit.prune <- prune.rpart(credit.rpart, cp= 0.013)
#plotting tree
plot(credit.prune)
text(credit.prune)

#in sample prediction
in_samp_tree <- predict(credit.rpart, type = "class") #in sample predcit
#table
table(credit_train$response,in_samp_tree,dnn = c("Truth","Predicted")) #table- 247,267,8,228
#misclassification rate
mean(ifelse(credit_train$response != in_samp_tree,1,0)) #0.084
#cost function
cost(credit_train$response,in_samp_tree) #0.4093


#out of sample prediction
out_samp_tree <- predict(credit.rpart,newdata =  credit_test, type = "class")
#table
table(credit_test$response,out_samp_tree, dnn = c("Truth","Predicted")) #89,97,3,61
#misclassification rate
mean(ifelse(credit_test$response != out_samp_tree,1,0)) #0.092
#cost function
cost(credit_test$response,out_samp_tree) #0.4480

#ROC
#in sample
credit.rpart.prob <- predict(credit.rpart,credit_train, type = "prob")
tree_pred_roc <- prediction(credit.rpart.prob[,2],credit_train$response)
tree_perf <- performance(tree_pred_roc,"tpr","fpr")
plot(tree_perf,colorize =TRUE)
as.numeric(performance(tree_pred_roc,'auc')@y.values) #AUC = 0.7542

#out of sample
credit.rpart.prob.out <- predict(credit.rpart,credit_test, type = "prob")
tree_pred_roc_out <- prediction(credit.rpart.prob.out[,2],credit_test$response)
tree_perf_out <- performance(tree_pred_roc_out,"tpr","fpr")
plot(tree_perf_out,colorize =TRUE)
as.numeric(performance(tree_pred_roc_out,'auc')@y.values) #AUC = 0.7454



#Discriminant Analysis
install.packages("MASS")
library(MASS)
credit_train$response <- as.factor(credit_train$response) #Y changed to factor
credit_lda <- lda(response~., data=credit_train)

#in sample prediction
prob_lda_insamp <- predict(credit_lda, data = credit_train)
pcut_lda <- 0.16
pred_lda_insamp <- (prob_lda_insamp$posterior[,2] >= pcut_lda)*1
table(credit_train$response,pred_lda_insamp, dnn = c("Obs","Pred")) #271,243,24,212
#misclassfication rate
mean(ifelse(credit_train$response != pred_lda_insamp,1,0))
#cost function - in sample
cost(credit_train$response,pred_lda_insamp) #0.484

#out of sample prediction
prob_lda_outsamp <- predict(credit_lda, newdata = credit_test)
pcut_out_lda <- 0.16
pred_lda_outsamp <- as.numeric(prob_lda_outsamp$posterior[,2] >= pcut_out_lda)
table(credit_test$response, pred_lda_outsamp, dnn = c("Obs","Pred")) #97,89,5,59

#misclassfication rate
mean(ifelse(credit_test$response != pred_lda_outsamp,1,0))
#cost function- out of sample
cost(credit_test$response,pred_lda_outsamp) #0.456

#ROC 
#in sample
roc_pred_train_lda <- prediction(prob_lda_insamp$posterior[,2],credit_train$response)
roc_perf_train_lda <- performance(roc_pred_train_lda,"tpr","fpr")
plot(roc_perf_train_lda, colorize = TRUE)
as.numeric(performance(roc_pred_train_lda,'auc')@y.values) #0.8251

#out of sample
roc_pred_test_lda <- prediction(prob_lda_outsamp$posterior[,2],credit_test$response)
roc_perf_test_lda <- performance(roc_pred_test_lda,"tpr","fpr")
plot(roc_perf_test_lda, colorize = TRUE)
as.numeric(performance(roc_pred_test_lda,'auc')@y.values) #0.8414


#Generalised Additive Model
install.packages("mgcv")
library(mgcv)
gam_credit <- gam(formula = response ~ chk_acct+ s(duration) + credit_his + purpose +
                    s(amount) + saving_acct + present_emp + installment_rate + sex + other_debtor + 
                    present_resid + property + s(age) + other_install + housing + n_credits +
                    job + n_people + telephone + foreign, family = binomial, data = credit_train)
summary(gam_credit)
plot(gam_credit, shade = TRUE, setWithMean= TRUE, scale = 0) 

#revised
gam_credit_final <- gam(formula = response ~ chk_acct+ duration + credit_his + purpose +
                          s(amount) + saving_acct + present_emp + installment_rate + sex + other_debtor + 
                          present_resid + property + age + other_install + housing + n_credits +
                          job + n_people + telephone + foreign, family = binomial, data = credit_train)
summary(gam_credit_final)

#in sample prediction
pcut_gam <- 0.16
prob_gam_in <- predict(gam_credit_final, credit_train, type = "response")
pred_gam_in <- (prob_gam_in >= pcut_gam)*1
#table
table(credit_train$response,pred_gam_in, dnn = c("Obs","Pred")) #270, 244,16,220
#cost
cost(credit_train$response,pred_gam_in) #0.432
mean(ifelse(credit_test$response != pred_gam_out,1,0))
#out of sample prediction
pcut_gam <- 0.16
prob_gam_out <- predict(gam_credit_final, credit_test, type = "response")
pred_gam_out <- (prob_gam_out >= pcut_gam)*1
#table
table(credit_test$response,pred_gam_out, dnn = c("Obs","Pred")) # 99,87,5,59
#cost
cost(credit_test$response,pred_gam_out) #0.448

#ROC
#in sample
roc_pred_train_gam <- prediction(as.numeric(prob_gam_in), credit_train$response)
roc_pref_train_gam <- performance(roc_pred_train_gam,"tpr","fpr")
plot(roc_pred_train_gam,colorize = TRUE)
as.numeric(performance(roc_pred_train_gam, 'auc')@y.values)

#out sample
roc_pred_test_gam <- prediction(as.numeric(prob_gam_out), credit_test$response)
roc_pref_test_gam <- performance(roc_pred_test_gam,"tpr","fpr")
plot(roc_pred_test_gam,colorize = TRUE)
as.numeric(performance(roc_pred_test_gam, 'auc')@y.values)


