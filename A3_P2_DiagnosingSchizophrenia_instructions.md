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
grouped_iqr <- aggregate(x = allData$IQR, by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(IQR=x)
grouped_sd <- aggregate(x = allData$sd, by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(sd=x)
grouped_pauseDuration <- aggregate(x = allData$pauseDuration, by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(pauseDuration=x)
grouped_propSpokenTime <- aggregate(x = allData$propSpokenTime, by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(propSpokenTime=x)
grouped_speechrate <- aggregate(x = allData$speechrate..nsyll.dur., by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(speechrate=x)
grouped_diagnosis <- aggregate(x = as.numeric(allData$diagnosis), by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(diagnosis=x)
grouped_study <- aggregate(x = as.numeric(allData$study), by = list(allData$uID), FUN = mean, na.rm=T) %>% dplyr::rename(study=x)


merged_grouped <- merge(grouped_iqr, grouped_pauseDuration, by="Group.1")
merged_grouped <- merge(merged_grouped, grouped_propSpokenTime, by="Group.1")
merged_grouped <- merge(merged_grouped, grouped_speechrate, by="Group.1")
merged_grouped <- merge(merged_grouped, grouped_sd, by="Group.1")
merged_grouped <- merge(merged_grouped, grouped_study, by="Group.1")
merged_grouped <- merge(merged_grouped, grouped_diagnosis, by="Group.1")

groupedData <- merged_grouped %>% dplyr::rename(uID=Group.1)

write_csv(merged_grouped, "groupedData.csv")


merged_grouped %>% 
  group_by(diagnosis) %>%
  summarize(participant=n(), 
            pauseDur = mean(pauseDuration, na.rm=T))
```

    ## # A tibble: 2 x 3
    ##   diagnosis participant pauseDur
    ##       <dbl>       <int>    <dbl>
    ## 1         0         131    0.918
    ## 2         1         134    1.14

``` r
# loading grouped data
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
         speechrate, propSpokenTime) %>% drop_na() # Pitch variablity model

groupedData2 <- groupedData %>% 
  select(uID, diagnosis, study, IQR, pauseDuration, 
         speechrate, sd) %>% drop_na() # Pause duration model

groupedData3 <- groupedData %>% 
  select(uID, diagnosis, study, IQR, pauseDuration, 
         propSpokenTime, sd) %>% drop_na() # Speech rate model

groupedData4 <- groupedData %>% 
  select(uID, diagnosis, study, IQR, 
         speechrate, propSpokenTime, sd) %>% drop_na() # Proportional of spoken time model

groupedData5 <-  groupedData %>% 
  select(uID, diagnosis, study, pauseDuration, 
         speechrate, propSpokenTime, sd) %>% drop_na() # Standard deviation model

groupedDataCombined <- groupedData %>% 
  select(uID, diagnosis, study, IQR, pauseDuration, 
         speechrate, propSpokenTime, sd) %>% drop_na() # The combined model
```

Running models & cross-validation
=================================

``` r
# partitioning the data using groupdata2
df_list <- partition(groupedDataSD, p = 0.2, cat_col = c("diagnosis",'study'), list_out = T)

 # defining the test-set and removing ID-column
#groupedPred <- df_list[[1]]

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


#  Investigating the metrics
metrics(test_results, truth = diagnosis, estimate = log_class) %>% 
  knitr::kable()
```

| .metric  | .estimator |  .estimate|
|:---------|:-----------|----------:|
| accuracy | binary     |  0.6136364|
| kap      | binary     |  0.2175732|

``` r
# extracting metrics of log_class
test_results %>%
  select(diagnosis, log_class, log_prob) %>%
  knitr::kable()
```

| diagnosis | log\_class |  log\_prob|
|:----------|:-----------|----------:|
| 0         | 1          |  0.5106467|
| 0         | 0          |  0.4569445|
| 0         | 0          |  0.3811560|
| 0         | 0          |  0.3688156|
| 0         | 1          |  0.5042692|
| 0         | 1          |  0.5313930|
| 0         | 1          |  0.5121303|
| 0         | 0          |  0.4221843|
| 0         | 0          |  0.3630443|
| 0         | 0          |  0.2165055|
| 0         | 0          |  0.0634569|
| 0         | 0          |  0.4895204|
| 0         | 0          |  0.3293245|
| 0         | 0          |  0.3933788|
| 0         | 1          |  0.5071054|
| 0         | 0          |  0.4214106|
| 0         | 0          |  0.4756275|
| 0         | 1          |  0.5163480|
| 0         | 0          |  0.4375860|
| 0         | 0          |  0.0481294|
| 0         | 0          |  0.2167004|
| 0         | 0          |  0.4848524|
| 0         | 0          |  0.4525687|
| 1         | 1          |  0.5080661|
| 1         | 1          |  0.5135907|
| 1         | 0          |  0.4626234|
| 1         | 0          |  0.2954272|
| 1         | 0          |  0.4935081|
| 1         | 1          |  0.5236607|
| 1         | 1          |  0.5338759|
| 1         | 0          |  0.3750225|
| 1         | 0          |  0.4491274|
| 1         | 1          |  0.5088235|
| 1         | 0          |  0.4960083|
| 1         | 1          |  0.5149235|
| 1         | 0          |  0.3795891|
| 1         | 0          |  0.3949503|
| 1         | 1          |  0.5279033|
| 1         | 1          |  0.5466571|
| 1         | 0          |  0.4914399|
| 1         | 0          |  0.4889315|
| 1         | 1          |  0.5065139|
| 1         | 1          |  0.5304413|
| 1         | 0          |  0.4605672|

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

#inspect performance metrics
 eval %>% 
  select(repeat_n = id, fold_n=id2,metric = .metric, estimate = .estimate) %>% 
  spread(metric, estimate) %>% 
  head() %>% 
  knitr::kable()
```

| repeat\_n | fold\_n |   accuracy|         kap|  mn\_log\_loss|   roc\_auc|
|:----------|:--------|----------:|-----------:|--------------:|----------:|
| Repeat1   | Fold1   |  0.5277778|   0.0496894|      0.7007131|  0.5263158|
| Repeat1   | Fold2   |  0.6388889|   0.2687500|      0.7576218|  0.5820433|
| Repeat1   | Fold3   |  0.6111111|   0.2050473|      0.8145216|  0.3900929|
| Repeat1   | Fold4   |  0.5000000|  -0.0156740|      0.7126635|  0.4613003|
| Repeat1   | Fold5   |  0.6176471|   0.2352941|      0.7497129|  0.7013889|
| Repeat2   | Fold1   |  0.5555556|   0.0971787|      0.7132563|  0.6037152|

Making a combined prediction from multiple models
=================================================

``` r
dataList = list(groupedData1, groupedData2, groupedData3,
            groupedData4, groupedData5, groupedDataCombined)

allPredictions <- data.frame()

for (df in dataList){
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
  
  if (NROW(allPredictions) < 1)  {
        allPredictions <- test_results
  } else {
    allPredictions <- cbind(allPredictions, test_results)
  }
}

absmax <- function(x) {x[which.max ( abs(x) )]}

ulala <- list()
  
for (i in 1:44){
  ulala[i] <- absmax(allPredictions[i, c(3, 6, 9, 12, 15, 18)])
}

ulala <- unlist(transpose(ulala))


absAllPredictions <- allPredictions
absAllPredictions[, c(3, 6, 9, 12, 15, 18)] <- abs(allPredictions[, c(3, 6, 9, 12, 15, 18)])

sumLogProb = absAllPredictions[,3]+absAllPredictions[,6]+absAllPredictions[,9]+absAllPredictions[,12]+absAllPredictions[,15]+absAllPredictions[,18]

final <- absAllPredictions[,2]*(absAllPredictions[,3]/sumLogProb)+absAllPredictions[,5]*(absAllPredictions[,6]/sumLogProb)+absAllPredictions[,8]*(absAllPredictions[,9]/sumLogProb)+absAllPredictions[,11]*(absAllPredictions[,12]/sumLogProb)+absAllPredictions[,14]*(absAllPredictions[,15]/sumLogProb)+absAllPredictions[,17]*(absAllPredictions[,18]/sumLogProb)

simplePred <- absAllPredictions[,2]+absAllPredictions[,5]+absAllPredictions[,8]+absAllPredictions[,11]+absAllPredictions[,14]+absAllPredictions[,17]


predData <- data.frame(predictions=final, actual=allPredictions[,1])
predData$predDiagnosis <- ifelse(predData$predictions < 0.5, "0", "1")
predData$predDiagnosis <- as.factor(predData$predDiagnosis)
predData$actual <- as.factor(predData$actual)

caret::confusionMatrix(predData$predDiagnosis, predData$actual, positive="0")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  0  1
    ##          0 20  6
    ##          1  3 15
    ##                                         
    ##                Accuracy : 0.7955        
    ##                  95% CI : (0.647, 0.902)
    ##     No Information Rate : 0.5227        
    ##     P-Value [Acc > NIR] : 0.0001702     
    ##                                         
    ##                   Kappa : 0.5875        
    ##                                         
    ##  Mcnemar's Test P-Value : 0.5049851     
    ##                                         
    ##             Sensitivity : 0.8696        
    ##             Specificity : 0.7143        
    ##          Pos Pred Value : 0.7692        
    ##          Neg Pred Value : 0.8333        
    ##              Prevalence : 0.5227        
    ##          Detection Rate : 0.4545        
    ##    Detection Prevalence : 0.5909        
    ##       Balanced Accuracy : 0.7919        
    ##                                         
    ##        'Positive' Class : 0             
    ## 

``` r
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
    ##          0 16  7
    ##          1  7 14
    ##                                           
    ##                Accuracy : 0.6818          
    ##                  95% CI : (0.5242, 0.8139)
    ##     No Information Rate : 0.5227          
    ##     P-Value [Acc > NIR] : 0.02387         
    ##                                           
    ##                   Kappa : 0.3623          
    ##                                           
    ##  Mcnemar's Test P-Value : 1.00000         
    ##                                           
    ##             Sensitivity : 0.6957          
    ##             Specificity : 0.6667          
    ##          Pos Pred Value : 0.6957          
    ##          Neg Pred Value : 0.6667          
    ##              Prevalence : 0.5227          
    ##          Detection Rate : 0.3636          
    ##    Detection Prevalence : 0.5227          
    ##       Balanced Accuracy : 0.6812          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

``` r
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
    ##          0 17  6
    ##          1  6 15
    ##                                           
    ##                Accuracy : 0.7273          
    ##                  95% CI : (0.5721, 0.8504)
    ##     No Information Rate : 0.5227          
    ##     P-Value [Acc > NIR] : 0.00455         
    ##                                           
    ##                   Kappa : 0.4534          
    ##                                           
    ##  Mcnemar's Test P-Value : 1.00000         
    ##                                           
    ##             Sensitivity : 0.7391          
    ##             Specificity : 0.7143          
    ##          Pos Pred Value : 0.7391          
    ##          Neg Pred Value : 0.7143          
    ##              Prevalence : 0.5227          
    ##          Detection Rate : 0.3864          
    ##    Detection Prevalence : 0.5227          
    ##       Balanced Accuracy : 0.7267          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

``` r
pacman::p_load(precrec)

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
model <- caretList(diagnosis~. , data=train_ensemble, trControl=control, methodList=algorithmList)
```

    ## Warning in trControlCheck(x = trControl, y = target): x$savePredictions ==
    ## TRUE is depreciated. Setting to 'final' instead.

    ## Warning in trControlCheck(x = trControl, y = target): indexes not defined
    ## in trControl. Attempting to set them ourselves, so each model in the
    ## ensemble will have the same resampling indexes.

``` r
results <- resamples(model)
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
    ## lda       0.5454545 0.6000000 0.6292929 0.6261616 0.6590909 0.7045455    0
    ## svmRadial 0.4772727 0.5000000 0.5280303 0.5357071 0.5722222 0.6222222    0
    ## rpart     0.3863636 0.5056818 0.5795455 0.5688384 0.6666667 0.6888889    0
    ## glm       0.5454545 0.6000000 0.6363636 0.6395960 0.6895202 0.7333333    0
    ## knn       0.4666667 0.4829545 0.5391414 0.5537879 0.5753788 0.7333333    0
    ## 
    ## Kappa 
    ##                  Min.     1st Qu.     Median       Mean   3rd Qu.
    ## lda        0.07756813  0.20235816 0.24968497 0.24802900 0.3184963
    ## svmRadial -0.07659574 -0.03325476 0.03267982 0.04760473 0.1339643
    ## rpart     -0.24789916 -0.02515600 0.12558610 0.11010172 0.3066206
    ## glm        0.07756813  0.20588235 0.26359510 0.27468856 0.3724021
    ## knn       -0.06508876 -0.04987851 0.07610939 0.10191439 0.1526640
    ##                Max. NA's
    ## lda       0.4041667    0
    ## svmRadial 0.2105263    0
    ## rpart     0.3558282    0
    ## glm       0.4674556    0
    ## knn       0.4610778    0

``` r
dotplot(results)
```

![](A3_P2_DiagnosingSchizophrenia_instructions_files/figure-markdown_github/unnamed-chunk-5-1.png)

``` r
# correlation between results
modelCor(results)
```

    ##                  lda svmRadial      rpart        glm        knn
    ## lda       1.00000000 0.4279265 0.08339108 0.94654327 0.15041767
    ## svmRadial 0.42792649 1.0000000 0.68068931 0.58317932 0.26781879
    ## rpart     0.08339108 0.6806893 1.00000000 0.19645303 0.15504092
    ## glm       0.94654327 0.5831793 0.19645303 1.00000000 0.09600059
    ## knn       0.15041767 0.2678188 0.15504092 0.09600059 1.00000000

``` r
splom(results)
```

![](A3_P2_DiagnosingSchizophrenia_instructions_files/figure-markdown_github/unnamed-chunk-5-2.png)

``` r
# stack using glm
stackControl <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
stack.glm <- caretStack(model, method="glm", metric="Accuracy", trControl=stackControl)
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
    ## Summary of sample sizes: 355, 355, 355, 355, 356, 356, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.6193948  0.2321098

``` r
# stack using random forest
stack.rf <- caretStack(model, method="rf", metric="Accuracy", trControl=stackControl)
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
    ## Summary of sample sizes: 355, 356, 355, 355, 355, 356, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##   2     0.6543029  0.3046427
    ##   3     0.6430286  0.2806765
    ##   5     0.6362232  0.2668724
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.
