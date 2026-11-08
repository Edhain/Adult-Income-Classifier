# ML Assignment 2: Classification Model Deployment

## a. Problem statement
The objective of this project is to build, evaluate, and deploy a machine learning classification model. The specific problem addressed here is predicting whether an individual's income exceeds $50K per year based on census demographic and employment data. This is a binary classification task designed to demonstrate end-to-end ML deployment workflow including modeling, evaluation, UI design, and deployment.

## b. Dataset description
* **Dataset Name:** Adult Income Dataset (from UCI/Kaggle public repository)
* **Target Variable:** `Target` (Income >50K or <=50K)
* **Features:** 14 features (Age, Workclass, fnlwgt, Education, Education-Num, Marital-Status, Occupation, Relationship, Race, Sex, Capital-Gain, Capital-Loss, Hours-per-week, Country)
* **Instance Size:** >30,000 instances (Comfortably meets the >500 instance requirement)
* **Description:** The dataset contains demographic, educational, and financial features to predict the income bracket of an individual.

## c. Github Repository Link
https://github.com/Edhain/Adult-Income-Classifier

## d. Models used
Below is the comparison table with the evaluation metrics calculated for all 5 implemented models on the test data:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8247 | 0.8568 | 0.7113 | 0.4596 | 0.5584 | 0.4722 |
| **Decision Tree** | 0.8131 | 0.7516 | 0.6083 | 0.6327 | 0.6203 | 0.4966 |
| **KNN** | 0.8353 | 0.8572 | 0.6786 | 0.6022 | 0.6381 | 0.5335 |
| **Naive Bayes** | 0.8093 | 0.8613 | 0.7064 | 0.3584 | 0.4755 | 0.4060 |
| **Support Vector Machine** | 0.8174 | 0.8518 | 0.8003 | 0.3240 | 0.4613 | 0.4299 |


## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves decent accuracy and AUC, but struggles significantly with Recall (0.4596), meaning it misses a large number of the positive (>50K) instances. |
| **Decision Tree** | Yields the highest Recall (0.6327) among all models, meaning it is best at identifying true positive instances, but suffers from the lowest overall AUC (0.7516). |
| **KNN** | Provides the best balanced performance overall. It achieved the highest Accuracy (0.8353), highest F1 Score (0.6381), and highest MCC (0.5335) out of the 5 models. |
| **Naive Bayes** | Ironically yields the highest AUC (0.8613) but suffers from the lowest Accuracy (0.8093) and a very poor Recall (0.3584). |
| **Support Vector Machine** | Extremely conservative in predicting the positive class. It has the highest Precision (0.8003) but the lowest Recall (0.3240), leading to a poor F1 score (0.4613). |
| **Overall, Winner for your dataset?** | **KNN (K-Nearest Neighbors)**. It provides the best balance between precision and recall, as evidenced by its leading F1 and MCC scores, while also topping the raw Accuracy metric. |