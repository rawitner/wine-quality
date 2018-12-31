Hackathon- Predicting Wine Quality
================
Rachel Witner
6/14/2018

For this hackathon experience, we were given a training dataset about wine quality and were tasked to predict wine quality for a test dataset. There were 11 numeric variables about the wine plus a quality indicator: 1 for good or 0 for bad. Below is what I did to come up with the most accurate prediction model, which ended up being 73-74% accurate from cross-validation, and 76% accurate as applied to the actual test data. :)

What the training data looks like:

|  fixed\_acidity|  volatile\_acidity|  citric\_acid|  sugar|  chlorides|  free\_sulf\_diox|  tot\_sulf\_diox|  density|    pH|  sulphates|  alcohol|  Quality|
|---------------:|------------------:|-------------:|------:|----------:|-----------------:|----------------:|--------:|-----:|----------:|--------:|--------:|
|             7.4|               0.70|          0.00|    1.9|      0.076|                11|               34|   0.9978|  3.51|       0.56|      9.4|        0|
|             7.8|               0.88|          0.00|    2.6|      0.098|                25|               67|   0.9968|  3.20|       0.68|      9.8|        0|
|             7.8|               0.76|          0.04|    2.3|      0.092|                15|               54|   0.9970|  3.26|       0.65|      9.8|        0|
|            11.2|               0.28|          0.56|    1.9|      0.075|                17|               60|   0.9980|  3.16|       0.58|      9.8|        1|
|             7.4|               0.70|          0.00|    1.9|      0.076|                11|               34|   0.9978|  3.51|       0.56|      9.4|        0|
|             7.4|               0.66|          0.00|    1.8|      0.075|                13|               40|   0.9978|  3.51|       0.56|      9.4|        0|

A correlation matrix to get an initial idea of the relationship between variables:

![](hackathon_files/figure-markdown_github/corrmatrix-1.png)

------------------------------------------------------------------------

Classification using GBM
------------------------

-   I found [this tutorial](https://amunategui.github.io/binary-outcome-modeling/) which uses a Generalized Boosting Model (GBM)
-   GBM is an ensemble of classification (or regression) trees-- it takes individual decision trees and aggregates them to form a better predictor than a single decision tree
-   It's contained in the `caret` package (**C**lassification **A**nd **Re**gression **T**raining) in R, which is a set of functions used for creating predictive models

``` r
outcomeName <- colnames(df)[12]
predictorsNames <- colnames(df)[1:11]

#change Quality to factor so gbm will use classification mode
df$Quality2 <- ifelse(df$Quality==1,'good','bad')
df$Quality2 <- as.factor(df$Quality2)
outcomeName <- 'Quality2'

#split into testing and training sets 
set.seed(1234)
splitIndex <- createDataPartition(df$Quality2, p = .75, list = FALSE, times = 1)
trainDF <- df[ splitIndex,]
testDF  <- df[-splitIndex,]
```

In this case, I cross-validated the data 3 times, therefore training it 3 times on different portions of the data before settling on the best tuning parameters. The `trainControl` function allows you to control the resampling of data. This will split the training data set internally and do it’s own train/test runs to figure out the best settings for the model.

``` r
objControl <- trainControl(method = 'cv', 
                           number = 3, 
                           returnResamp = 'none', 
                           summaryFunction = twoClassSummary, 
                           classProbs = TRUE)
```

`train` sets up a grid of tuning parameters for a number of classification and regression routines, fits each model and calculates a resampling based performance measure.

I fit the classification model using ROC instead of RMSE since its a binary classification.

``` r
objModel <- train(trainDF[,predictorsNames], trainDF$Quality2, 
                  method = 'gbm', 
                  trControl = objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.3642             nan     0.1000    0.0086
    ##      2        1.3416             nan     0.1000    0.0100
    ##      3        1.3227             nan     0.1000    0.0094
    ##      4        1.3064             nan     0.1000    0.0058
    ##      5        1.2846             nan     0.1000    0.0110
    ##      6        1.2695             nan     0.1000    0.0059
    ##      7        1.2541             nan     0.1000    0.0054
    ##      8        1.2369             nan     0.1000    0.0072
    ##      9        1.2232             nan     0.1000    0.0029
    ##     10        1.2098             nan     0.1000    0.0048
    ##     20        1.1162             nan     0.1000    0.0006
    ##     40        1.0378             nan     0.1000   -0.0009
    ##     60        0.9850             nan     0.1000   -0.0020
    ##     80        0.9423             nan     0.1000   -0.0007
    ##    100        0.9115             nan     0.1000   -0.0007
    ##    120        0.8822             nan     0.1000   -0.0011
    ##    140        0.8577             nan     0.1000   -0.0019
    ##    150        0.8471             nan     0.1000   -0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.3403             nan     0.1000    0.0201
    ##      2        1.3045             nan     0.1000    0.0113
    ##      3        1.2763             nan     0.1000    0.0124
    ##      4        1.2464             nan     0.1000    0.0090
    ##      5        1.2198             nan     0.1000    0.0094
    ##      6        1.1991             nan     0.1000    0.0075
    ##      7        1.1783             nan     0.1000    0.0088
    ##      8        1.1592             nan     0.1000    0.0055
    ##      9        1.1407             nan     0.1000    0.0048
    ##     10        1.1241             nan     0.1000    0.0046
    ##     20        1.0308             nan     0.1000    0.0001
    ##     40        0.9205             nan     0.1000   -0.0018
    ##     60        0.8469             nan     0.1000   -0.0013
    ##     80        0.7812             nan     0.1000   -0.0004
    ##    100        0.7289             nan     0.1000   -0.0020
    ##    120        0.6843             nan     0.1000   -0.0011
    ##    140        0.6491             nan     0.1000   -0.0022
    ##    150        0.6292             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.3340             nan     0.1000    0.0195
    ##      2        1.2956             nan     0.1000    0.0154
    ##      3        1.2569             nan     0.1000    0.0127
    ##      4        1.2278             nan     0.1000    0.0085
    ##      5        1.1976             nan     0.1000    0.0102
    ##      6        1.1748             nan     0.1000    0.0040
    ##      7        1.1493             nan     0.1000    0.0109
    ##      8        1.1276             nan     0.1000    0.0088
    ##      9        1.1101             nan     0.1000    0.0043
    ##     10        1.0941             nan     0.1000    0.0014
    ##     20        0.9712             nan     0.1000   -0.0004
    ##     40        0.8250             nan     0.1000    0.0008
    ##     60        0.7401             nan     0.1000   -0.0022
    ##     80        0.6684             nan     0.1000   -0.0028
    ##    100        0.6047             nan     0.1000   -0.0003
    ##    120        0.5521             nan     0.1000   -0.0020
    ##    140        0.5037             nan     0.1000   -0.0003
    ##    150        0.4826             nan     0.1000   -0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.3554             nan     0.1000    0.0124
    ##      2        1.3321             nan     0.1000    0.0094
    ##      3        1.3088             nan     0.1000    0.0096
    ##      4        1.2916             nan     0.1000    0.0031
    ##      5        1.2789             nan     0.1000    0.0054
    ##      6        1.2582             nan     0.1000    0.0041
    ##      7        1.2429             nan     0.1000    0.0059
    ##      8        1.2334             nan     0.1000    0.0032
    ##      9        1.2192             nan     0.1000    0.0057
    ##     10        1.2072             nan     0.1000    0.0049
    ##     20        1.1171             nan     0.1000    0.0010
    ##     40        1.0276             nan     0.1000    0.0006
    ##     60        0.9797             nan     0.1000    0.0001
    ##     80        0.9484             nan     0.1000   -0.0004
    ##    100        0.9176             nan     0.1000   -0.0011
    ##    120        0.8930             nan     0.1000   -0.0023
    ##    140        0.8694             nan     0.1000   -0.0013
    ##    150        0.8571             nan     0.1000   -0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.3465             nan     0.1000    0.0145
    ##      2        1.3148             nan     0.1000    0.0129
    ##      3        1.2846             nan     0.1000    0.0129
    ##      4        1.2615             nan     0.1000    0.0054
    ##      5        1.2340             nan     0.1000    0.0134
    ##      6        1.2153             nan     0.1000    0.0056
    ##      7        1.1919             nan     0.1000    0.0088
    ##      8        1.1705             nan     0.1000    0.0098
    ##      9        1.1523             nan     0.1000    0.0061
    ##     10        1.1392             nan     0.1000    0.0024
    ##     20        1.0337             nan     0.1000   -0.0002
    ##     40        0.9167             nan     0.1000   -0.0018
    ##     60        0.8529             nan     0.1000   -0.0008
    ##     80        0.7979             nan     0.1000    0.0001
    ##    100        0.7558             nan     0.1000   -0.0031
    ##    120        0.7186             nan     0.1000   -0.0019
    ##    140        0.6770             nan     0.1000   -0.0036
    ##    150        0.6614             nan     0.1000   -0.0041
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.3436             nan     0.1000    0.0177
    ##      2        1.2979             nan     0.1000    0.0143
    ##      3        1.2627             nan     0.1000    0.0118
    ##      4        1.2306             nan     0.1000    0.0134
    ##      5        1.2061             nan     0.1000    0.0090
    ##      6        1.1843             nan     0.1000    0.0049
    ##      7        1.1528             nan     0.1000    0.0132
    ##      8        1.1298             nan     0.1000    0.0079
    ##      9        1.1095             nan     0.1000    0.0042
    ##     10        1.0924             nan     0.1000    0.0038
    ##     20        0.9627             nan     0.1000    0.0008
    ##     40        0.8170             nan     0.1000   -0.0018
    ##     60        0.7434             nan     0.1000   -0.0019
    ##     80        0.6761             nan     0.1000   -0.0034
    ##    100        0.6213             nan     0.1000   -0.0037
    ##    120        0.5679             nan     0.1000   -0.0009
    ##    140        0.5291             nan     0.1000   -0.0018
    ##    150        0.5064             nan     0.1000   -0.0011
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.3508             nan     0.1000    0.0139
    ##      2        1.3259             nan     0.1000    0.0091
    ##      3        1.3055             nan     0.1000    0.0076
    ##      4        1.2878             nan     0.1000    0.0071
    ##      5        1.2657             nan     0.1000    0.0082
    ##      6        1.2482             nan     0.1000    0.0075
    ##      7        1.2341             nan     0.1000    0.0055
    ##      8        1.2186             nan     0.1000    0.0039
    ##      9        1.2082             nan     0.1000    0.0024
    ##     10        1.1944             nan     0.1000    0.0072
    ##     20        1.0924             nan     0.1000   -0.0004
    ##     40        0.9822             nan     0.1000    0.0007
    ##     60        0.9211             nan     0.1000   -0.0000
    ##     80        0.8773             nan     0.1000   -0.0031
    ##    100        0.8481             nan     0.1000   -0.0012
    ##    120        0.8261             nan     0.1000   -0.0011
    ##    140        0.8035             nan     0.1000   -0.0011
    ##    150        0.7906             nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.3372             nan     0.1000    0.0177
    ##      2        1.2907             nan     0.1000    0.0206
    ##      3        1.2545             nan     0.1000    0.0130
    ##      4        1.2206             nan     0.1000    0.0147
    ##      5        1.1936             nan     0.1000    0.0078
    ##      6        1.1712             nan     0.1000    0.0095
    ##      7        1.1498             nan     0.1000    0.0065
    ##      8        1.1358             nan     0.1000    0.0016
    ##      9        1.1173             nan     0.1000    0.0041
    ##     10        1.1041             nan     0.1000    0.0025
    ##     20        0.9764             nan     0.1000    0.0028
    ##     40        0.8529             nan     0.1000    0.0007
    ##     60        0.7803             nan     0.1000   -0.0025
    ##     80        0.7274             nan     0.1000   -0.0035
    ##    100        0.6885             nan     0.1000   -0.0023
    ##    120        0.6463             nan     0.1000   -0.0031
    ##    140        0.6054             nan     0.1000   -0.0016
    ##    150        0.5820             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.3285             nan     0.1000    0.0236
    ##      2        1.2787             nan     0.1000    0.0201
    ##      3        1.2396             nan     0.1000    0.0123
    ##      4        1.2005             nan     0.1000    0.0192
    ##      5        1.1689             nan     0.1000    0.0135
    ##      6        1.1385             nan     0.1000    0.0080
    ##      7        1.1079             nan     0.1000    0.0087
    ##      8        1.0842             nan     0.1000    0.0072
    ##      9        1.0681             nan     0.1000    0.0035
    ##     10        1.0460             nan     0.1000    0.0080
    ##     20        0.9103             nan     0.1000    0.0004
    ##     40        0.7671             nan     0.1000   -0.0003
    ##     60        0.6716             nan     0.1000   -0.0020
    ##     80        0.6016             nan     0.1000   -0.0023
    ##    100        0.5343             nan     0.1000   -0.0009
    ##    120        0.4814             nan     0.1000   -0.0002
    ##    140        0.4370             nan     0.1000   -0.0018
    ##    150        0.4196             nan     0.1000   -0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.3275             nan     0.1000    0.0186
    ##      2        1.2966             nan     0.1000    0.0095
    ##      3        1.2669             nan     0.1000    0.0105
    ##      4        1.2355             nan     0.1000    0.0149
    ##      5        1.2097             nan     0.1000    0.0110
    ##      6        1.1800             nan     0.1000    0.0102
    ##      7        1.1539             nan     0.1000    0.0094
    ##      8        1.1336             nan     0.1000    0.0061
    ##      9        1.1130             nan     0.1000    0.0061
    ##     10        1.0968             nan     0.1000    0.0051
    ##     20        0.9783             nan     0.1000    0.0021
    ##     40        0.8648             nan     0.1000   -0.0007
    ##     60        0.7878             nan     0.1000   -0.0019
    ##     80        0.7230             nan     0.1000   -0.0018
    ##    100        0.6667             nan     0.1000    0.0005

Summary of the relative influence of each variable: ![](hackathon_files/figure-markdown_github/rel_influence-1.png)

|                   | var               |    rel.inf|
|-------------------|:------------------|----------:|
| alcohol           | alcohol           |  21.698265|
| tot\_sulf\_diox   | tot\_sulf\_diox   |  15.643432|
| sulphates         | sulphates         |  14.208490|
| volatile\_acidity | volatile\_acidity |  12.711506|
| pH                | pH                |   6.738208|
| citric\_acid      | citric\_acid      |   5.779274|
| fixed\_acidity    | fixed\_acidity    |   5.199710|
| free\_sulf\_diox  | free\_sulf\_diox  |   5.007050|
| chlorides         | chlorides         |   4.860197|
| density           | density           |   4.635480|
| sugar             | sugar             |   3.518388|

-   We can see that alcohol, total sulfur dioxide, sulphates, and volatile acidity had the largest impact on the quality
-   Now need to figure out what tuning parameters were most important to the model:

``` r
print(objModel)
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 600 samples
    ##  11 predictor
    ##   2 classes: 'bad', 'good' 
    ## 
    ## Pre-processing: centered (11), scaled (11) 
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 401, 399, 400 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  ROC        Sens       Spec     
    ##   1                   50      0.8038238  0.7743490  0.6832533
    ##   1                  100      0.8175546  0.7868395  0.6867612
    ##   1                  150      0.8200815  0.7743490  0.6903836
    ##   2                   50      0.8196064  0.7774349  0.7116983
    ##   2                  100      0.8270160  0.7805502  0.7189430
    ##   2                  150      0.8297563  0.7806383  0.7118127
    ##   3                   50      0.8384551  0.7930700  0.7330130
    ##   3                  100      0.8417926  0.8087345  0.7118508
    ##   3                  150      0.8382989  0.7930700  0.7296195
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## ROC was used to select the optimal model using the largest value.
    ## The final values used for the model were n.trees = 100,
    ##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

-   There are three main tuning parameters for the gbm model:
    -   number of iterations, i.e. trees (n.trees)
    -   complexity of the tree (interaction.depth)
    -   learning rate: how quickly the algorithm adapts (shrinkage)
    -   the minimum number of training set samples in a node to commence splitting (n.minobsinnode)
-   ROC is the area under the ROC curve. An area of 1 represents a model that made all predictions perfectly. An area of 0.5 represents a model as good as random.
-   There is a tradeoff between sensitivity and specificity:
    -   sensitivity is the true positive rate (aka recall)
    -   specificity is the true negative rate
-   A perfect predictor would be 100% sensitive and 100% specific

-   Now that the model is trained, call the predict function on the trained model and testing data

``` r
predictions <- predict(object = objModel, 
                       testDF[, predictorsNames], 
                       type='raw')
head(predictions, n = 15)
```

    ##  [1] bad  bad  bad  bad  good bad  bad  bad  bad  bad  good bad  bad  bad 
    ## [15] bad 
    ## Levels: bad good

-   Let's see how accurate the model was for our generated testing data

``` r
print(postResample(pred = predictions, 
                   obs = as.factor(testDF$Quality2)))
```

    ##  Accuracy     Kappa 
    ## 0.7386935 0.4723639

-   Around 74% accuracy-- could be better, but I can live with that
-   Lastly, I produced the final predictions for the test data

``` r
realpredictions <- predict(object=objModel, df_test_file[,predictorsNames], type='raw')
realpredictions <- ifelse(realpredictions=="good",1,0)
head(realpredictions)
```

    ## [1] 0 0 0 1 0 0

``` r
write.table(realpredictions, "predictions.txt", sep="\t", row.names=FALSE, col.names=FALSE)
```
