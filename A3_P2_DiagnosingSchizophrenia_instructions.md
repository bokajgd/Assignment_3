Assignment 3 - Part 2 - Diagnosing schizophrenia from voice
-----------------------------------------------------------

In the previous part of the assignment you generated a bunch of "features", that is, of quantitative descriptors of voice in schizophrenia. We then looked at whether we could replicate results from the previous literature. We now want to know whether we can automatically diagnose schizophrenia from voice only, that is, relying on the set of features you produced last time, we will try to produce an automated classifier. Again, remember that the dataset containst 7 studies and 3 languages. Feel free to only include Danish (Study 1-4) if you feel that adds too much complexity.

Issues to be discussed your report: - Should you run the analysis on all languages/studies at the same time? - Choose your best acoustic feature from part 1. How well can you diagnose schizophrenia just using it? - Identify the best combination of acoustic features to diagnose schizophrenia using logistic regression. - Discuss the "classification" process: which methods are you using? Which confounds should you be aware of? What are the strength and limitation of the analysis? - Bonus question: Logistic regression is only one of many classification algorithms. Try using others and compare performance. Some examples: Discriminant Function, Random Forest, Support Vector Machine, etc. The package caret provides them. - Bonus Bonus question: It is possible combine the output of multiple classification models to improve classification accuracy. For inspiration see, <https://machinelearningmastery.com/machine-learning-ensembles-with-r/> The interested reader might also want to look up 'The BigChaos Solution to the Netflix Grand Prize'

Learning objectives
-------------------

-   Learn the basics of classification in a machine learning framework
-   Design, fit and report logistic regressions
-   Apply feature selection techniques

### Let's start

We first want to build a logistic regression to see whether you can diagnose schizophrenia from your best acoustic feature. Let's use the full dataset and calculate the different performance measures (accuracy, sensitivity, specificity, PPV, NPV, ROC curve). You need to think carefully as to how we should (or not) use study and subject ID.

Then cross-validate the logistic regression and re-calculate performance on the testing folds. N.B. The cross-validation functions you already have should be tweaked: you need to calculate these new performance measures. Alternatively, the groupdata2 and cvms package created by Ludvig are an easy solution.

N.B. the predict() function generates log odds (the full scale between minus and plus infinity). Log odds &gt; 0 indicates a choice of 1, below a choice of 0. N.N.B. you need to decide whether calculate performance on each single test fold or save all the prediction for test folds in one datase, so to calculate overall performance. N.N.N.B. Now you have two levels of structure: subject and study. Should this impact your cross-validation? N.N.N.N.B. A more advanced solution could rely on the tidymodels set of packages (warning: Time-consuming to learn as the documentation is sparse, but totally worth it)

``` r
# Making a dataframe grouped by participant 
# Grouping each variable by participant
grouped_iqr <- aggregate(x = allData$IQR, by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(IQR=x)
grouped_sd <- aggregate(x = allData$sd, by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(sd=x)
grouped_pauseDuration <- aggregate(x = allData$pauseDuration, by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(pauseDuration=x)
grouped_propSpokenTime <- aggregate(x = allData$propSpokenTime, by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(propSpokenTime=x)
grouped_speechrate <- aggregate(x = allData$speechrate..nsyll.dur., by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(speechrate=x)
grouped_diagnosis <- aggregate(x = as.numeric(allData$diagnosis), by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(diagnosis=x)
grouped_study <- aggregate(x = as.numeric(allData$study), by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(study=x)


# Merged grouped data frames
merged_grouped <- merge(grouped_iqr, grouped_pauseDuration, by="Group.1")
merged_grouped <- merge(merged_grouped, grouped_propSpokenTime, by="Group.1")
merged_grouped <- merge(merged_grouped, grouped_speechrate, by="Group.1")
merged_grouped <- merge(merged_grouped, grouped_sd, by="Group.1")
merged_grouped <- merge(merged_grouped, grouped_study, by="Group.1")
merged_grouped <- merge(merged_grouped, grouped_diagnosis, by="Group.1")

# Changing Group.1 collumn name to uID
groupedData <- merged_grouped %>% dplyr::rename(uID=Group.1)

# Saving grouped data as csv file
write_csv(merged_grouped, "groupedData.csv")

# Making diagnosis factors
groupedData$diagnosis <- as.factor(groupedData$diagnosis) 
```

Defining models with grouped data (1 data point pr. participant instead of approx. 10)
======================================================================================

``` r
# creating dataframes including relevant variables
groupedDataIQR <- groupedData %>% 
  select(uID, diagnosis, study, IQR) %>% drop_na() # Pitch variablity model

groupedDataPause <- groupedData %>% 
  select(uID, diagnosis, study, pauseDuration) %>% drop_na() # Pause duration model

groupedDataSpeechrate <- groupedData %>% 
  select(uID, diagnosis, study, speechrate) %>% drop_na() # Speech rate model

groupedDataSpokentime <- groupedData %>% 
  select(uID, diagnosis, study, propSpokenTime) %>% drop_na() # Proportional of spoken time model

groupedDataSD <-  groupedData %>% 
  select(uID, diagnosis, study, sd) %>% drop_na() # Standard deviation model

groupedData1 <- groupedData %>% 
  select(uID, diagnosis, study, IQR, pauseDuration, 
         speechrate, propSpokenTime) %>% drop_na() 

groupedData2 <- groupedData %>% 
  select(uID, diagnosis, study, IQR, pauseDuration, 
         speechrate, sd) %>% drop_na() 

groupedData3 <- groupedData %>% 
  select(uID, diagnosis, study, IQR, pauseDuration, 
         propSpokenTime, sd) %>% drop_na() 

groupedData4 <- groupedData %>% 
  select(uID, diagnosis, study, IQR, 
         speechrate, propSpokenTime, sd) %>% drop_na() 

groupedData5 <-  groupedData %>% 
  select(uID, diagnosis, study, pauseDuration, 
         speechrate, propSpokenTime, sd) %>% drop_na() 

groupedDataCombined <- groupedData %>% 
  select(uID, diagnosis, study, IQR, pauseDuration, 
         speechrate, propSpokenTime, sd) %>% drop_na() # The combined model
```

Running models & cross-validation
=================================

``` r
# partitioning the data using groupdata2 by diagnosis and study as cat_cols
df_list <- partition(groupedDataSD, p = 0.2, cat_col = c("diagnosis",'study'), list_out = T)

 # defining the test-set and removing ID and study column
df_test <- df_list[[1]]
df_test <- df_test %>% 
  select(-uID, -study)

# defining the train-set and removing ID-column
df_train <- df_list[[2]]
df_train <- df_train %>% 
  select(-uID, -study)



# Defining the recipe for train-data
# We've removed NAs so no need to check for missing values
rec <- df_train %>% recipe(diagnosis ~ .) %>% # defines outcome of pre-processing
  step_center(all_numeric()) %>% # centering all numeric values
  step_scale(all_numeric()) %>% # scaling all numeric values
  step_corr(all_numeric()) %>% # corr testing all predictors
  prep(training = df_train) # defining the train-set

# extracting 'df_train' from rec_train (the recipe)
train_baked <- juice(rec)
test_baked <- rec %>% bake(df_test)


# defining model as logistic regression
log_fit <- 
  logistic_reg() %>% 
  set_mode("classification") %>% 
  set_engine("glm") %>% 
  fit(diagnosis ~ ., data = train_baked)

# defining model as a support-vector-machine
svm_fit <- 
  svm_rbf() %>% 
  set_mode("classification") %>% 
  set_engine("kernlab") %>%
  fit(diagnosis ~ ., data = train_baked)



# investigating both logistic regression and the SVM-models
# get multiple at once
test_results <- 
  test_baked %>% 
  select(diagnosis) %>% 
  mutate(
    log_class = predict(log_fit, new_data = test_baked) %>% 
      pull(.pred_class),
    log_prob  = predict(log_fit, new_data = test_baked, type = "prob") %>% 
      pull(.pred_1),
    svm_class = predict(svm_fit, new_data = test_baked) %>% 
      pull(.pred_class),
    svm_prob  = predict(svm_fit, new_data = test_baked, type = "prob") %>% 
      pull(.pred_1)
  )


# Investigating the metrics
metrics(test_results, truth = diagnosis, estimate = log_class) %>% 
  knitr::kable()
```

| .metric  | .estimator |  .estimate|
|:---------|:-----------|----------:|
| accuracy | binary     |  0.7045455|
| kap      | binary     |  0.4016736|

``` r
# extracting metrics of log_class
test_results %>%
  select(diagnosis, log_class, log_prob) %>%
  knitr::kable()
```

| diagnosis | log\_class |  log\_prob|
|:----------|:-----------|----------:|
| 0         | 0          |  0.4958873|
| 0         | 0          |  0.3675619|
| 0         | 0          |  0.4817821|
| 0         | 0          |  0.4220216|
| 0         | 1          |  0.5403227|
| 0         | 0          |  0.4728338|
| 0         | 0          |  0.4626617|
| 0         | 0          |  0.4660929|
| 0         | 0          |  0.3719763|
| 0         | 0          |  0.3973734|
| 0         | 0          |  0.4971041|
| 0         | 1          |  0.5408487|
| 0         | 0          |  0.3816077|
| 0         | 0          |  0.0548760|
| 0         | 0          |  0.4929751|
| 0         | 0          |  0.4967371|
| 0         | 0          |  0.4238137|
| 0         | 0          |  0.4992490|
| 0         | 0          |  0.0763235|
| 0         | 0          |  0.4982388|
| 0         | 1          |  0.5347254|
| 0         | 1          |  0.5143241|
| 0         | 0          |  0.4892894|
| 1         | 1          |  0.5165617|
| 1         | 0          |  0.4870423|
| 1         | 1          |  0.5178817|
| 1         | 1          |  0.5205197|
| 1         | 1          |  0.5224622|
| 1         | 0          |  0.4679508|
| 1         | 1          |  0.5447778|
| 1         | 1          |  0.5374731|
| 1         | 0          |  0.3658777|
| 1         | 0          |  0.2895620|
| 1         | 1          |  0.5470162|
| 1         | 0          |  0.4913054|
| 1         | 1          |  0.5454181|
| 1         | 0          |  0.4490454|
| 1         | 0          |  0.4781931|
| 1         | 0          |  0.4391757|
| 1         | 1          |  0.5537050|
| 1         | 0          |  0.4961072|
| 1         | 1          |  0.5588305|
| 1         | 1          |  0.5495607|
| 1         | 1          |  0.5374467|

``` r
# # Plotting area-under-the-curve (ROC-curve)
# test_results %>%
#   roc_curve(truth = diagnosis, log_prob) %>%
#   autoplot()

# Cross validation of grouped data
cv_folds <- vfold_cv(df_train, v = 5, repeats = 2, strata = diagnosis, group = uID)

#prepare data set and fetch train data
cv_folds <- cv_folds %>% 
  mutate(recipes = splits %>%
           # prepper is a wrapper for `prep()` which handles `split` objects
           map(prepper, recipe = rec),
         train_data = splits %>% map(training))

# train model of each fold
  # create a non-fitted model
log_fit <- 
  logistic_reg() %>%
  set_mode("classification") %>% 
  set_engine("glm")

cv_folds <- cv_folds %>%  mutate(
  log_fits = pmap(list(recipes, train_data), #input 
                            ~ fit(log_fit, formula(.x), data = bake(object = .x, new_data = .y)) # function to apply
                 ))

predict_log <- function(split, rec, model) {
  # IN
    # split: a split data
    # rec: recipe to prepare the data
    # 
  # OUT
    # a tibble of the actual and predicted results
  baked_test <- bake(rec, testing(split))
  tibble(
    actual = baked_test$diagnosis,
    predicted = predict(model, new_data = baked_test) %>% pull(.pred_class),
    prop_sui =  predict(model, new_data = baked_test, type = "prob") %>% pull(.pred_1),
    prop_non_sui =  predict(model, new_data = baked_test, type = "prob") %>% pull(`.pred_0`)
  ) 
}

# apply our function to each split, which their respective recipes and models (in this case log fits) and save it to a new col
cv_folds <- cv_folds %>% 
  mutate(pred = pmap(list(splits, recipes, log_fits) , predict_log))


eval <- cv_folds %>% 
  mutate(
    metrics = pmap(list(pred), ~ metrics(., truth = actual, estimate = predicted, prop_sui))) %>% 
  select(id, id2, metrics) %>% 
  unnest(metrics)

# inspect performance metrics
 eval %>% 
  select(repeat_n = id, fold_n=id2,metric = .metric, estimate = .estimate) %>% 
  spread(metric, estimate) %>% 
  head() %>% 
  knitr::kable()
```

| repeat\_n | fold\_n |   accuracy|         kap|  mn\_log\_loss|   roc\_auc|
|:----------|:--------|----------:|-----------:|--------------:|----------:|
| Repeat1   | Fold1   |  0.4444444|  -0.1009174|      0.7105334|  0.5232198|
| Repeat1   | Fold2   |  0.5833333|   0.1768293|      0.7189577|  0.6222910|
| Repeat1   | Fold3   |  0.4444444|  -0.1356467|      0.7776755|  0.5758514|
| Repeat1   | Fold4   |  0.7222222|   0.4321767|      0.7868509|  0.7894737|
| Repeat1   | Fold5   |  0.5588235|   0.1114983|      0.7842492|  0.5243056|
| Repeat2   | Fold1   |  0.5833333|   0.1509434|      0.7678335|  0.5975232|

Making a combined prediction from multiple models
=================================================

``` r
# List of multiple models
dataList = list(groupedData1, groupedData2, groupedData3,
            groupedData4, groupedData5, groupedDataCombined)

# empty data frame
allPredictions <- data.frame()


# Creating loop to run through list of models
for (df in dataList){ #Partitioning data
  df_list <- partition(df, p = 0.2, cat_col = c("diagnosis", "study"), list_out = T)
  
  # define test-set
  df_test <- df_list[[1]]
  df_test <- df_test %>% 
  select(-uID, -study)

  # define train-set
  df_train <- df_list[[2]]
  df_train <- df_train %>% 
  select(-uID, -study)
  
  # Defining the recipe for train-data
  # We've removed NAs so no need to check for missing values
  rec <- df_train %>% recipe(diagnosis ~ .) %>% # defines outcome of pre-processing
    step_center(all_numeric()) %>% # centering all numeric values
    step_scale(all_numeric()) %>% # scaling all numeric values
    step_corr(all_numeric()) %>% 
    prep(training = df_train) # defining the train-set

  # extracting 'df_train' from rec_train (the recipe)
  train_baked <- juice(rec)
  test_baked <- rec %>% bake(df_test)


  # defining model as logistic regression
  log_fit <- 
    logistic_reg() %>% 
    set_mode("classification") %>% 
    set_engine("glm") %>% 
    fit(diagnosis ~ ., data = train_baked)
  
  # defining model as a support-vector-machine
  svm_fit <- 
    svm_rbf() %>% 
    set_mode("classification") %>% 
    set_engine("kernlab") %>%
    fit(diagnosis ~ ., data = train_baked)
  
  # getting test-results
  test_results <- test_baked %>% 
    select(diagnosis) %>% 
    mutate(
      log_class = predict(log_fit, new_data = test_baked) %>% 
        pull(.pred_class),
      log_prob  = predict(log_fit, new_data = test_baked, type = "prob") %>% 
        pull(.pred_1),
      svm_class = predict(svm_fit, new_data = test_baked) %>% 
        pull(.pred_class),
      svm_prob  = predict(svm_fit, new_data = test_baked, type = "prob") %>% 
        pull(.pred_1)
    )
  
    #  Investigating the metrics
  metrics(test_results, truth = diagnosis, estimate = log_class) %>% 
    knitr::kable()
  
  # extracting metrics of log_class
  test_results <- test_results %>%
    select(diagnosis, log_class, log_prob) %>% 
    mutate(log_prob = log_prob-0.5,
           diagnosis = as.numeric(diagnosis)-1,
           log_class = as.numeric(log_class)-1) 
  
  if (NROW(allPredictions) < 1)  { #saving test results into a dataframe
        allPredictions <- test_results
  } else {
    allPredictions <- cbind(allPredictions, test_results)
  }
}

# Function for extracting max deviation from 0.5
absmax <- function(x) {x[which.max ( abs(x) )]}

# creating empty list
ulala <- list()
  
# add max deviation value of probabilty of prediction between the five models from each participant into a list
for (i in 1:44){
  ulala[i] <- absmax(allPredictions[i, c(3, 6, 9, 12, 15, 18)])
}

# transposing list into a vector
ulala <- unlist(transpose(ulala))

# Make new data fram with absolute values of deviations from 0.5
absAllPredictions <- allPredictions
absAllPredictions[, c(3, 6, 9, 12, 15, 18)] <- abs(allPredictions[, c(3, 6, 9, 12, 15, 18)])

# Summing probability outputs for each prediction from each model for each participant
sumLogProb = absAllPredictions[,3]+absAllPredictions[,6]+absAllPredictions[,9]+absAllPredictions[,12]+absAllPredictions[,15]+absAllPredictions[,18]

# Calculating weighted sum of probabilities
final <- absAllPredictions[,2]*(absAllPredictions[,3]/sumLogProb)+absAllPredictions[,5]*(absAllPredictions[,6]/sumLogProb)+absAllPredictions[,8]*(absAllPredictions[,9]/sumLogProb)+absAllPredictions[,11]*(absAllPredictions[,12]/sumLogProb)+absAllPredictions[,14]*(absAllPredictions[,15]/sumLogProb)+absAllPredictions[,17]*(absAllPredictions[,18]/sumLogProb)

# Adding predictions from all 6 models together
simplePred <- absAllPredictions[,2]+absAllPredictions[,5]+absAllPredictions[,8]+absAllPredictions[,11]+absAllPredictions[,14]+absAllPredictions[,17]

# Adding 'weighted sum' probs and actual predictions to a data frame
predData <- data.frame(predictions=final, actual=allPredictions[,1])
# Turning weighted sum probabilities into diagnosis predictions
predData$predDiagnosis <- ifelse(predData$predictions < 0.5, "0", "1")
predData$predDiagnosis <- as.factor(predData$predDiagnosis)
predData$actual <- as.factor(predData$actual)

# Making confusion matrix
caret::confusionMatrix(predData$predDiagnosis, predData$actual, positive="0")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  0  1
    ##          0 21  7
    ##          1  2 14
    ##                                         
    ##                Accuracy : 0.7955        
    ##                  95% CI : (0.647, 0.902)
    ##     No Information Rate : 0.5227        
    ##     P-Value [Acc > NIR] : 0.0001702     
    ##                                         
    ##                   Kappa : 0.5858        
    ##                                         
    ##  Mcnemar's Test P-Value : 0.1824224     
    ##                                         
    ##             Sensitivity : 0.9130        
    ##             Specificity : 0.6667        
    ##          Pos Pred Value : 0.7500        
    ##          Neg Pred Value : 0.8750        
    ##              Prevalence : 0.5227        
    ##          Detection Rate : 0.4773        
    ##    Detection Prevalence : 0.6364        
    ##       Balanced Accuracy : 0.7899        
    ##                                         
    ##        'Positive' Class : 0             
    ## 

``` r
# Same for max prob method
predData2 <- data.frame(predictions=ulala, actual=allPredictions[,1])
predData2$predDiagnosis <- ifelse(predData2$predictions < 0, "0", "1")
predData2$predDiagnosis <- as.factor(predData2$predDiagnosis)
predData2$actual <- as.factor(predData2$actual)

caret::confusionMatrix(predData2$predDiagnosis, predData2$actual, positive="0")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  0  1
    ##          0 21  6
    ##          1  2 15
    ##                                           
    ##                Accuracy : 0.8182          
    ##                  95% CI : (0.6729, 0.9181)
    ##     No Information Rate : 0.5227          
    ##     P-Value [Acc > NIR] : 4.451e-05       
    ##                                           
    ##                   Kappa : 0.6326          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.2888          
    ##                                           
    ##             Sensitivity : 0.9130          
    ##             Specificity : 0.7143          
    ##          Pos Pred Value : 0.7778          
    ##          Neg Pred Value : 0.8824          
    ##              Prevalence : 0.5227          
    ##          Detection Rate : 0.4773          
    ##    Detection Prevalence : 0.6136          
    ##       Balanced Accuracy : 0.8137          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

``` r
#Same for un-weighted sum method
predData3 <- data.frame(predictions=simplePred, actual=allPredictions[,1])
predData3$predDiagnosis <- ifelse(predData3$predictions < 3 , "0", "1")
predData3$predDiagnosis <- as.factor(predData3$predDiagnosis)
predData3$actual <- as.factor(predData3$actual)

caret::confusionMatrix(predData3$predDiagnosis, predData3$actual, positive="0")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  0  1
    ##          0 15  7
    ##          1  8 14
    ##                                           
    ##                Accuracy : 0.6591          
    ##                  95% CI : (0.5008, 0.7951)
    ##     No Information Rate : 0.5227          
    ##     P-Value [Acc > NIR] : 0.04748         
    ##                                           
    ##                   Kappa : 0.3182          
    ##                                           
    ##  Mcnemar's Test P-Value : 1.00000         
    ##                                           
    ##             Sensitivity : 0.6522          
    ##             Specificity : 0.6667          
    ##          Pos Pred Value : 0.6818          
    ##          Neg Pred Value : 0.6364          
    ##              Prevalence : 0.5227          
    ##          Detection Rate : 0.3409          
    ##    Detection Prevalence : 0.5000          
    ##       Balanced Accuracy : 0.6594          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

``` r
# Combining all three methods
prediction_df <- data.frame(predWeight=as.numeric(predData$predDiagnosis)-1, predMax=as.numeric(predData2$predDiagnosis)-1, predSum=as.numeric(predData3$predDiagnosis)-1, actual=allPredictions[,1])
prediction_df$predDiagnosis <- ifelse(prediction_df$predWeight+prediction_df$predMax+prediction_df$predSum < 2, "0", "1")
prediction_df$predDiagnosis <- as.factor(prediction_df$predDiagnosis)
prediction_df$actual <- as.factor(prediction_df$actual)

caret::confusionMatrix(prediction_df$predDiagnosis, prediction_df$actual, positive="0")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  0  1
    ##          0 20  7
    ##          1  3 14
    ##                                           
    ##                Accuracy : 0.7727          
    ##                  95% CI : (0.6216, 0.8853)
    ##     No Information Rate : 0.5227          
    ##     P-Value [Acc > NIR] : 0.0005717       
    ##                                           
    ##                   Kappa : 0.5407          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.3427817       
    ##                                           
    ##             Sensitivity : 0.8696          
    ##             Specificity : 0.6667          
    ##          Pos Pred Value : 0.7407          
    ##          Neg Pred Value : 0.8235          
    ##              Prevalence : 0.5227          
    ##          Detection Rate : 0.4545          
    ##    Detection Prevalence : 0.6136          
    ##       Balanced Accuracy : 0.7681          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

``` r
pacman::p_load(precrec)

# Plotting roc-curve for weighted sum method 
precrec_obj <- evalmod(scores = as.numeric(predData$predDiagnosis)-1, labels = as.numeric(predData$actual)-1)
autoplot(precrec_obj)
```

![](A3_P2_DiagnosingSchizophrenia_instructions_files/figure-markdown_github/unnamed-chunk-4-1.png)

``` r
citation("precrec")
```

    ## 
    ## To cite precrec in publications use:
    ## 
    ##   Takaya Saito and Marc Rehmsmeier (2017). Precrec: fast and
    ##   accurate precision-recall and ROC curve calculations in R.
    ##   Bioinformatics (2017) 33 (1): 145-147. doi:
    ##   10.1093/bioinformatics/btw570
    ## 
    ## A BibTeX entry for LaTeX users is
    ## 
    ##   @Article{,
    ##     title = {Precrec: fast and accurate precision-recall and ROC curve calculations in R},
    ##     author = {Takaya Saito and Marc Rehmsmeier},
    ##     journal = {Bioinformatics},
    ##     year = {2017},
    ##     volume = {33 (1)},
    ##     pages = {145-147},
    ##     doi = {10.1093/bioinformatics/btw570},
    ##   }

``` r
# Attempting to stack multiple methods
p_load(caretEnsemble, caret, randomForest)
train_ensemble <- groupedData %>% select(-uID, -study)
train_ensemble <- na.omit(train_ensemble)
train_ensemble$diagnosis <- as.factor(train_ensemble$diagnosis)
str(train_ensemble)
```

    ## 'data.frame':    222 obs. of  6 variables:
    ##  $ IQR           : num  20.6 20.1 21.8 50.5 14.8 ...
    ##  $ pauseDuration : num  0.939 0.693 0.942 0.98 0.906 ...
    ##  $ propSpokenTime: num  0.58 0.723 0.628 0.596 0.623 ...
    ##  $ speechrate    : num  2.74 3.93 3.54 2.92 2.99 ...
    ##  $ sd            : num  17.7 16.7 20.7 35.1 12.8 ...
    ##  $ diagnosis     : Factor w/ 2 levels "0","1": 1 1 1 2 1 2 1 2 1 2 ...
    ##  - attr(*, "na.action")= 'omit' Named int  2 4 122 124 125 129 132 133 134 135 ...
    ##   ..- attr(*, "names")= chr  "2" "4" "122" "124" ...

``` r
levels(train_ensemble$diagnosis) <- c('control', 'schizo')

control <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda','svmRadial','rpart', 'glm', 'knn')
modls <- caretList(diagnosis~. , data=train_ensemble, trControl=control, methodList=algorithmList)
```

    ## Warning in trControlCheck(x = trControl, y = target): x$savePredictions ==
    ## TRUE is depreciated. Setting to 'final' instead.

    ## Warning in trControlCheck(x = trControl, y = target): indexes not defined
    ## in trControl. Attempting to set them ourselves, so each model in the
    ## ensemble will have the same resampling indexes.

``` r
results <- resamples(modls)
summary(results)
```

    ## 
    ## Call:
    ## summary.resamples(object = results)
    ## 
    ## Models: lda, svmRadial, rpart, glm, knn 
    ## Number of resamples: 10 
    ## 
    ## Accuracy 
    ##                Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## lda       0.5000000 0.6034091 0.6250000 0.6258081 0.6554293 0.7111111    0
    ## svmRadial 0.3777778 0.5333333 0.5454545 0.5363131 0.5753788 0.5909091    0
    ## rpart     0.5227273 0.5587121 0.5729798 0.5765657 0.5944444 0.6363636    0
    ## glm       0.5000000 0.5965909 0.6136364 0.6214141 0.6555556 0.7111111    0
    ## knn       0.4772727 0.5000000 0.5169192 0.5473737 0.5888889 0.6590909    0
    ## 
    ## Kappa 
    ##                   Min.     1st Qu.     Median       Mean   3rd Qu.
    ## lda       -0.002070393  0.19453704 0.25074977 0.24602250 0.3089956
    ## svmRadial -0.257485030  0.01063830 0.07946165 0.05314910 0.1365213
    ## rpart      0.049676026  0.06820390 0.10671034 0.11999453 0.1491554
    ## glm       -0.002070393  0.19453704 0.22683682 0.23625857 0.2883824
    ## knn       -0.049792531 -0.01043841 0.03361993 0.08825655 0.1655093
    ##                Max. NA's
    ## lda       0.4036697    0
    ## svmRadial 0.1835052    0
    ## rpart     0.2494670    0
    ## glm       0.4179104    0
    ## knn       0.3209877    0

``` r
dotplot(results)
```

![](A3_P2_DiagnosingSchizophrenia_instructions_files/figure-markdown_github/unnamed-chunk-5-1.png)

``` r
# correlation between results
modelCor(results)
```

    ##                    lda    svmRadial      rpart         glm         knn
    ## lda        1.000000000 -0.008523634 -0.1003781  0.78905959 -0.06223151
    ## svmRadial -0.008523634  1.000000000  0.1795750 -0.32699091 -0.17769376
    ## rpart     -0.100378149  0.179574976  1.0000000 -0.11278240  0.65996158
    ## glm        0.789059594 -0.326990911 -0.1127824  1.00000000  0.01478323
    ## knn       -0.062231505 -0.177693765  0.6599616  0.01478323  1.00000000

``` r
splom(results)
```

![](A3_P2_DiagnosingSchizophrenia_instructions_files/figure-markdown_github/unnamed-chunk-5-2.png)

``` r
# stack using glm
stackControl <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
stack.glm <- caretStack(modls, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)
```

    ## A glm ensemble of 2 base models: lda, svmRadial, rpart, glm, knn
    ## 
    ## Ensemble results:
    ## Generalized Linear Model 
    ## 
    ## 444 samples
    ##   5 predictor
    ##   2 classes: 'control', 'schizo' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 2 times) 
    ## Summary of sample sizes: 355, 355, 356, 355, 355, 355, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.6408069  0.2735855

``` r
# stack using random forest
stack.rf <- caretStack(modls, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.rf)
```

    ## A rf ensemble of 2 base models: lda, svmRadial, rpart, glm, knn
    ## 
    ## Ensemble results:
    ## Random Forest 
    ## 
    ## 444 samples
    ##   5 predictor
    ##   2 classes: 'control', 'schizo' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 2 times) 
    ## Summary of sample sizes: 355, 355, 355, 355, 356, 355, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##   2     0.6249872  0.2444390
    ##   3     0.6249617  0.2450966
    ##   5     0.6125383  0.2211685
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.
