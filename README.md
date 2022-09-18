# kNN Classifier 

kNN or k-Nearest Neighbours Classifier is a very simple and easy to understand machine learning algorithm. In this repository, I have build a k Nearest Neighbours classifier to classify the patients suffering from Breast Cancer.

So, let's get started.

<a class="anchor" id="0.1"></a>
# **Table of Contents**


1. [Introduction to k Nearest Neighbours Algorithm](#1)
2. [Intuition of kNN](#2)
3. [How to decide the number of neighbours in kNN](#3)
4. [Eager learners vs lazy learners](#4)
5. [Import libraries](#5)
6. [Import dataset](#6)
7. [Exploratory data analysis](#7)
8. [Data visualization](#8)
9. [Declare feature vector and target variable](#9)
10. [Split data into separate training and test set](#10)
11.	[Feature engineering](#11)
12.	[Feature scaling](#12)
13.	[Fit Neighbours classifier to the training set](#13)
14.	[Predict the test-set results](#14)
15.	[Check the accuracy score](#15)
16.	[Rebuild kNN classification model using different values of k](#16)
17.	[Confusion matrix](#17)
18.	[Classification matrices](#18)
19.	[ROC - AUC](#19)
20.	[k-Fold Cross Validation](#20)
21.	[Results and conclusion](#21)

# **1. Introduction to k Nearest Neighbours algorithm** <a class="anchor" id="1"></a>

[Table of Contents](#0.1)

In machine learning, k Nearest Neighbours or kNN is the simplest of all machine learning algorithms. It is a non-parametric algorithm used for classification and regression tasks. Non-parametric means there is no assumption required for data distribution. So, kNN does not require any underlying assumption to be made. In both classification and regression tasks, the input consists of the k closest training examples in the feature space. The output depends upon whether kNN is used for classification or regression purposes.

-	In kNN classification, the output is a class membership. The given data point is classified based on the majority of type of its neighbours. The data point is assigned to the most frequent class among its k nearest neighbours. Usually k is a small positive integer. If k=1, then the data point is simply assigned to the class of that single nearest neighbour.

-	In kNN regression, the output is simply some property value for the object. This value is the average of the values of k nearest neighbours.


kNN is a type of `instance-based learning` or `lazy learning`. **Lazy learning** means it does not require any training data points for model generation. All training data will be used in the testing phase. This makes training faster and testing slower and costlier. So, the testing phase requires more time and memory resources.

In kNN, the neighbours are taken from a set of objects for which the class or the object property value is known. This can be thought of as the training set for the kNN algorithm, though no explicit training step is required. In both classification and regression kNN algorithm, we can assign weight to the contributions of the neighbours. So, nearest neighbours contribute more to the average than the more distant ones.

# **2. Intuition of kNN** <a class="anchor" id="2"></a>

[Table of Contents](#0.1)

The kNN algorithm intuition is very simple to understand. It simply calculates the distance between a sample data point and all the other training data points. The distance can be Euclidean distance or Manhattan distance. Then, it selects the k nearest data points where k can be any integer. Finally, it assigns the sample data point to the class to which the majority of the k data points belong.

Now, we will see kNN algorithm in action. Suppose, we have a dataset with two variables which are classified as `Red` and `Blue`.

In kNN algorithm, k is the number of nearest neighbours. Generally, k is an odd number because it helps to decide the majority of the class. When k=1, then the algorithm is known as the nearest neighbour algorithm.

Now, we want to classify a new data point `X` into `Blue` class or `Red` class. Suppose the value of k is 3. The kNN algorithm starts by calculating the distance between `X` and all the other data points. It then finds the 3 nearest points with least distance to point `X`. 

In the final step of the kNN algorithm, we assign the new data point `X` to the majority of the class of the 3 nearest points. If 2 of the 3 nearest points belong to the class `Red` while 1 belong to the class `Blue`, then we classify the new data point  as `Red`.

# **3. How to decide the number of neighbours in kNN** <a class="anchor" id="3"></a>

[Table of Contents](#0.1)

While building the kNN classifier model, one question that come to my mind is what should be the value of nearest neighbours (k) that yields highest accuracy. This is a very important question because the classification accuracy depends upon our choice of k.

The number of neighbours (k) in kNN is a parameter that we need to select at the time of model building. Selecting the optimal value of k in kNN is the most critical problem. A small value of k means that noise will have higher influence on the result. So, probability of overfitting is very high. A large value of k makes it computationally expensive in terms of time to build the kNN model. Also, a large value of k will have a smoother decision boundary which means lower variance but higher bias.

The data scientists choose an odd value of k if the number of classes is even. We can apply the elbow method to select the value of k. To optimize the results, we can use Cross Validation technique. Using the cross-validation technique, we can test the kNN algorithm with different values of k. The model which gives good accuracy can be considered to be an optimal choice. It depends on individual cases and at times best process is to run through each possible value of k and test our result.

# **4. Eager learners vs lazy learners** <a class="anchor" id="4"></a>

[Table of Contents](#0.1)

Eager learners mean when giving training data points, we will construct a generalized model before performing prediction on given new points to classify. We can think of such learners as being ready, active and eager to classify new data points.

Lazy learning means there is no need for learning or training of the model and all of the data points are used at the time of prediction. Lazy learners wait until the last minute before classifying any data point. They merely store the training dataset and waits until classification needs to perform. Lazy learners are also known as instance-based learners because lazy learners store the training points or instances, and all learning is based on instances.

Unlike eager learners, lazy learners do less work in the training phase and more work in the testing phase to make a classification.

# **5. Import libraries** <a class="anchor" id="5"></a>

[Table of Contents](#0.1)

Importing pandas, numpy, seaborn

# **6. Import dataset** <a class="anchor" id="6"></a>

[Table of Contents](#0.1)

We are using the Breast Cancer Wisconsin dataset

# **7. Exploratory data analysis** <a class="anchor" id="7"></a>

[Table of Contents](#0.1)

Now, I will explore the data to gain insights about the data. 

# **8. Data Visualization** <a class="anchor" id="8"></a>

[Table of Contents](#0.1)

Now, we have a basic understanding of our data. I will supplement it with some data visualization to get better understanding
of our data. We can see that all the variables in the dataset are positively skewed.

### Interpretation for Univariate Feature Plot:

- The correlation coefficient ranges from -1 to +1. 
- When it is close to +1, this signifies that there is a strong positive correlation. So, we can see that there is a strong positive correlation between `Class` and `Bare_Nuclei`, `Class` and `Uniformity_Cell_Shape`, `Class` and `Uniformity_Cell_Size`.
- When it is clsoe to -1, it means that there is a strong negative correlation. When it is close to 0, it means that there is no correlation. 
- We can see that all the variables are positively correlated with `Class` variable. Some variables are strongly positive correlated while some variables are negatively correlated.

### Interpretation of Multivariate Plots
![image](https://user-images.githubusercontent.com/35486320/190885715-a9cb1947-0930-4aa0-add4-ac49936e157c.png)

From the above correlation heat map, we can conclude that :-

1. `Class` is highly positive correlated with `Uniformity_Cell_Size`, `Uniformity_Cell_Shape` and `Bare_Nuclei`. (correlation coefficient = 0.82).
2. `Class` is positively correlated with `Clump_thickness`(correlation coefficient=0.72), `Marginal_Adhesion`(correlation coefficient=0.70), `Single_Epithelial_Cell_Size)`(correlation coefficient = 0.68) and `Normal_Nucleoli`(correlation coefficient=0.71).
3. `Class` is weekly positive correlated with `Mitoses`(correlation coefficient=0.42).
4. The `Mitoses` variable is weekly positive correlated with all the other variables(correlation coefficient < 0.50).


# **9. Declare feature vector and target variable** <a class="anchor" id="9"></a>

[Table of Contents](#0.1)

# **10. Split data into separate training and test set** <a class="anchor" id="10"></a>

[Table of Contents](#0.1)

# **11. Feature Engineering** <a class="anchor" id="11"></a>

[Table of Contents](#0.1)

**Feature Engineering** is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power. I will carry out feature engineering on different types of variables.

We now have training and testing set ready for model building. Before that, we should map all the feature variables onto the same scale. It is called `feature scaling`.

# **12. Feature Scaling** <a class="anchor" id="12"></a>

[Table of Contents](#0.1)

# **13. Fit K Neighbours Classifier to the training eet** <a class="anchor" id="13"></a>

[Table of Contents](#0.1)

Using KNeighborsClassifier from sklearn.neighbours, taking k = 3, we fit the data using X_train and y_train

# **14. Predict test-set results** <a class="anchor" id="14"></a>

[Table of Contents](#0.1)

Using predict.proba for calculating the probability of getting an outcome 

# **15. Check accuracy score** <a class="anchor" id="15"></a>

[Table of Contents](#0.1)

I will calculate the null accuracy by calculating the total number of rows of both classes and dividing the total with the highest number of entries of a target class

# **16. Rebuild kNN Classification model using different values of k** <a class="anchor" id="16"></a>

[Table of Contents](#0.1)

I have build the kNN classification model using k=3. Now, I will increase the value of k and see its effect on accuracy.

### Interpretation

- Our original model accuracy score with k=3 is `0.9714`. Now, we can see that we get same accuracy score of `0.9714` with k=5. But, if we increase the value of k further, this would result in enhanced accuracy.
- With k=6,7,8 we get accuracy score of `0.9786`. So, it results in performance improvement.
- If we increase k to 9, then accuracy decreases again to `0.9714`.

# **17. Confusion matrix** <a class="anchor" id="17"></a>

[Table of Contents](#0.1)

A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.

Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-

- **True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.

- **True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.

- **False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**

- **False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**

These four outcomes are summarized in a confusion matrix given below.

# **18. Classification metrices** <a class="anchor" id="18"></a>

[Table of Contents](#0.1)

**Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. I have described these terms in later. We can print a classification report.

# **19. ROC-AUC** <a class="anchor" id="19"></a>

[Table of Contents](#0.1)

Another tool to measure the classification model performance visually is **ROC Curve**. ROC Curve stands for **Receiver Operating Characteristic Curve**. An **ROC Curve** is a plot which shows the performance of a classification model at various  classification threshold levels. 

# **20. k-fold Cross Validation** <a class="anchor" id="20"></a>

[Table of Contents](#0.1)

In this section, I will apply k-fold Cross Validation technique to improve the model performance. Cross-validation is a statistical method of evaluating generalization performance It is more stable and thorough than using a train-test split to evaluate model performance. 
