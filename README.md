# Impact-of-Accuracy-Fairness-Tradeoff-in-Machine-Learning-Models
As learning models have become more advanced, concerns about fairness have become more prominent. There are many techniques in machine learning to eliminate the bias in the model and generate a model which gives predictions that are fair and accurate. The most prevalent technique in fair machine learning is to integrate fairness as a constraint or penalization term in the prediction loss minimization, which eventually limits the information provided to decision-makers. 
The main purpose of this project is to study the effect of regularization on the accuracy-fairness trade-off. This project will investigate the inherent bias of algorithms as well as evaluate fairness approaches devised to reduce prejudice, with an emphasis on accuracy deterioration, if any.

# Datasets Used in the Project
For evaluating the effects of regularization on accuracy and fairness two datasets namely Bank Marketing and Adult datasets are used. Both the datasets are taken from UCI Machine Learning Repository. 
1. Bank Dataset
    The information in the data relates to direct marketing activities run by a Portuguese bank. The dataset has 21 variables that are a mixture of categorical, ordinal,     and numerical data types such as age, education, loan, housing, job, etc. There are a total of 41,1161 rows of data, and 10,614 with missing values, leaving 30,547       complete rows. There are two class values, where age > =25 and age <25, meaning it is a binary classification task. in the dataset represents a person and if the         person has subscribed ('yes') or not ('no') to a bank term deposit.
    
2. Adult Dataset
   The dataset involves personal details such as education level to predict whether an individual will earn more or less than $50,000 per year.The dataset provides        variables that are a mixture of categorical, ordinal, and numerical data types such as Age, Education, Age, Sex, Race, Occupation, etc., There are a total of          48,842 rows of data, and 3,620 with missing values, leaving 45,222 complete rows. There are two class values, where income > 50 K and income <=50 K, meaning it is a    binary classification task. 

# Classification Model
There are various classification algorithms in machine learning like logistic regression, multilayer perceptron, support vector machines, etc. In this Project, the 
Support Vector Machine algorithm is used for classification. Support Vector Machine algorithm finds the best margin that separates the classes, thus reducing the risk of error on the dataset.A Standard SVM model with 5-fold cross-validation is used to determine the maximum accuracy and fairness of both datasets The scikit-learn
library is used for the SVM model. Each section/fold of a given dataset is used as a testing set at some stage There are five sections to the data set. The first fold is used to test the model in the first iteration, while the others are used to train it. In the second iteration, the second fold is used as the testing set, while the remaining folds are used as the training set. This process is repeated until each of the five folds has been evaluated.


![support-vector-machine-algorithm](https://user-images.githubusercontent.com/103538049/200060524-ccebe7bf-81c4-416b-9f1e-13defc6d14a4.png)  




![grid_search_cross_validation](https://user-images.githubusercontent.com/103538049/200061007-83b9438e-e292-4431-bfb8-ace7dbc2f6e1.png)


# Fainness Based Model
Unfairness can be either direct or indirect. Direct unfairness happens when a protected trait causes an adverse result directly, whereas indirect unfairness results 
from other factors that could be used to proxy the protected characteristic. There are two sources of unfairness in supervised machine learning. To begin with, machine learning predictions are trained on data that may have inherent biases. As a result, by learning from biased or prejudiced targets, typical learning procedures' 
prediction outcomes are unlikely to be fair. Second, even if the targets are fair, the learning process may compromise fairness because the goal of machine learning 
is to create the most accurate predictions.The fact that models are essentially based on data meansthat they will generally reflect the biases found in the data, 
very often increasing them. 

Even though biases are a problem, they could be mitigated or even eliminated. There are various ways to evaluate how fair a model is, but in this investigation, the "Equality of Opportunity Difference" is the central focus. This measure examines the ratio of groups that were positively categorized as protected versus unprotected. when everyone has the same opportunities. Fairness algorithms can aid in the reduction of bias during pre-processing, in-processing, or post-processing. There are many fairness algorithms that can be used for reducing bias like Adversarial debiasing, and Reweighing. In this investigation, Reweighing is utilized to mitigate the bias. 

# Objectives 
The investigation is divided into three major tasks
1. Analyse whether or not better generalization could corresspond to fairer models.
   In this task, SVM is used with 5-fold cross validation on training dataset and hyperparameters are vaired to selected a model with highest accuracy and a model with    highest fairness.
2. Apply a fairness-based algorithm and analyse the impact on accuracy and fairness
   In this task, fairness based-algorithm reweighing is used with SVM and 5-fold cross validation and hyperparameters are vaired tp select a model with highest            accuracy and highest fairness. 
3. Based on the above results suggest a model selection startegy that accounts for both acuuracy and fairness. 

# Results and Observations. 

Task 1

A few selected values of C and gamma were provided to the SVM model, where C = 0.001, 0.01, 0.1, 1, 10 and 100 and gamma = 0.001, 0.01, 0.1, 1, and 10. Table below displays the results of the most accurate and fair models on training and testing sets for both datasets.The bank Marketing dataset is already quite fair as compared to the Adult dataset. Although, when evaluatingon the test set both the models for the bank dataset showed almost the same accuracy, interestingly the fairness was 
lower in the second (most Fair) model. For the Adult dataset, the test result shows some difference in fairness as compared to the cross-validation results. The fairer model of banks, as well as the Adult dataset, showed a better TRPD in comparison with training models. 

![image](https://user-images.githubusercontent.com/103538049/200065975-0180e4cb-ee2c-4887-ae4c-bbd9a7f4c445.png)


The plots of hyperparameter variations with respect to hyperparameters and dataset are available below

![image](https://user-images.githubusercontent.com/103538049/200066256-5003840b-2094-4c87-98b0-920966ee62e6.png)


![image](https://user-images.githubusercontent.com/103538049/200066318-d2333a23-a73e-4bcd-975c-d1430170e115.png)


![image](https://user-images.githubusercontent.com/103538049/200066387-4eb17c6b-1e76-4527-95e3-b0d5a943d1e9.png)


![image](https://user-images.githubusercontent.com/103538049/200066447-363455bd-343c-4214-a2a9-e668a58568eb.png)



Task 2
The reweighing algorithm was applied during the preprocessing stage to improve the fairness of the models. The original training and testing data for both the datasets were passed to the algorithm and the output of the algorithm with the weights generated in the process was then passed to the SVM classifier with 5-fold cross-validation. The results obtained for task 2 are shown in table below.


![image](https://user-images.githubusercontent.com/103538049/200066755-c6b9710e-8b19-4bdc-9d8f-b51a72555759.png)


In the bank dataset, after applying reweighing interestingly the accuracy did not change to a great extent. There is only a little drop in accuracy but the reweighing made both the models quite fair as compared to the models in Task 1. Whereas, the adult dataset showed a significant decrease in accuracy and an increase in fairness in both the models, improvements in fairness were observed in both the datasets and both the models (Most Accurate and Most Fair).


Task 3
In this task, we have to create a strategy that will choose the best model that accounts for both accuracy as well as fairness. For the selection process, a scoring system is created. To come up with the most optimal hyperparameters, it is necessary to understand if the model is more skewed towards accuracy or fairness. To evaluate the hyperparameters, a scoring system is created. The results in after Task 1 and Task 2 were transformed into a table. The table includes the Hyperparameters, accuracy score and fairness scores. For Giving the scores to accuracy all the combinations of hyperparameters are taken into consideration with their accuracy scores. The model with highest accuracy is given a score of 10. The range between highest and lowest accuracy is split into 10 equal parts and the remaining models get the score between 1 and 10 depending on the accuracy. For the fairness metric, the model with the lowest TPR difference is given the score of 10. Again, the range between lowest and highest fairness metric is divided into 10 equal parts and the remaining model get a score between 1 to 10. After calculating all the scores, the scores of accuracies and fairness are simply added together to give a total score. The scoring system generated is then transformed into DataFrame for better visualizations. The DataFrames for both the datasets and both the models can be seen in Figures below.

![image](https://user-images.githubusercontent.com/103538049/200067529-61a9d996-6d79-4d85-8d6f-da1df7631c8e.png)



![image](https://user-images.githubusercontent.com/103538049/200067746-02ef70e0-7418-4381-8048-62703ed84c7b.png)



![image](https://user-images.githubusercontent.com/103538049/200067898-b111a386-a52f-4d1a-9fa6-dac214a03088.png)



![image](https://user-images.githubusercontent.com/103538049/200067952-15474007-d7d9-42c7-b985-9a83f12e24bb.png)



The table below shows the results of applying scoring system to both datasets. 

![image](https://user-images.githubusercontent.com/103538049/200068310-f052ce2a-2d70-46c7-9451-c523505ed8ff.png)


From the above we can see that the bank dataset shows exactly same accuracy and fairness for both the models. But interestingly when we compare the results of bank dataset models with Task 1 and Task 2 (Table 1 and table2), we can see that the hyperparameters the model selection strategy selected are a combination of hyperparameters of Task 1 and Task 2 models. We can also see that the fairness looks quite balanced. For the Adult dataset shows similar results on accuracy butfairness metrics is better than task 2. This shows that by utilizing the combinations from Task 1 and Task 2, we can create a fairer model with good accuracy. 


