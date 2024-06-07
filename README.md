# Project_4
Airline Satisfaction Survey Prediction Machine Learning Project 
Objective of the Project:  
To analyze and predict airline passenger satisfaction based on various features  using various machine learning models. By exploring and comparing different algorithms, we aim to identify key factors that influence passenger satisfaction and determine the most accurate model for predicting satisfaction levels. 
 
## Data Source:
For this project, we used data files in CSV format from Kaggle website. 
Content
Gender: Gender of the passengers (Female, Male)
Customer Type: The customer type (Loyal customer, disloyal customer)
Age: The actual age of the passengers
Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)
Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)
Flight distance: The flight distance of this journey
Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)
Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient
Ease of Online booking: Satisfaction level of online booking
Gate location: Satisfaction level of Gate location
Food and drink: Satisfaction level of Food and drink
Online boarding: Satisfaction level of online boarding
Seat comfort: Satisfaction level of Seat comfort
Inflight entertainment: Satisfaction level of inflight entertainment
On-board service: Satisfaction level of On-board service
Leg room service: Satisfaction level of Leg room service
Baggage handling: Satisfaction level of baggage handling
Check-in service: Satisfaction level of Check-in service
Inflight service: Satisfaction level of inflight service
Cleanliness: Satisfaction level of Cleanliness
Departure Delay in Minutes: Minutes delayed when departure
Arrival Delay in Minutes: Minutes delayed when Arrival
Satisfaction: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)

Link for the same is provided below:
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?select=train.csv



## Instructions:
•    A new repository  called project_4 was created, cloned and pushed to GitHub.
The instructions for this project are divided into the following subsections:
1.    Preprocess the Data
2.    Compile, Train, and Evaluate the Model
3.    Optimize the Model
4.    Write a report

Step 1: Preprocess the Data
Data Uplaod:  
Two CSV files  named train.csv and test.csv from kaggle website were uploaded, read and merged into a Pandas dataframe for analysis in Google Colab.
Data Cleaning : 
Null values were found and removed 
No duplicates were found 
Unnecessary columns ‘Unnamed: 1’ and ‘id’ were dropped 
Converted categorical features to numerical by using `pd.get_dummies()` 
Feature Selection:
Target Variable (y): ‘satisfaction’ (1 for satisfied, 0 for neutral/dissatisfied). 
Features Variables (X): 
'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Gender', 'Customer Type', 'Type of Travel', 'Class' 
 
Step 2: Compile, Train, and Evaluate the Model
The following models were built, trained, and compared to predict passenger satisfaction. 
•    Random Forest Classifier 
•    Logistic Regression 
•    Support Vector Classifier 
•    K Nearest Neighbors Classifier 
•    Naive Bayes Classifier 
•    Decision Tree 
•    Gradient Boosting Classifier 
•    XBG Classifier 
 
Applied four techniques for model evaluation:
1)    Scaling: Applied Standard Scaler. Create a new Google Colab file and name it Satisfaction.ipynb
2)    Without applying Scaling. Create a new Google Colab file and name it No_Scaling_ Satisfaction.ipynb
3)    Removing Outliers: Used IQR technique to detect and remove outliers.  Create a new Google Colab file and name it Outlier_Removal_ Satisfaction.ipynb
4)    Log Transformation: Used np.log1p(). Create a new Google Colab file and name it Log_Transform_Satisfaction.ipynb
 
Data Splitting: Used `train_test_split()` to divide data into training and testing sets. 
After splitting , each model is fiited by using model.fit  (X_train and y_train) 
The predictions are saved for the testing data labels by using  model.predict(X_test)   
Evaluated all the models using appropriate metrics such as accuracy score, classification report(precision, recall, f1 score), confusion matrix, and ROC-AUC. 
Compared the performance of the different models to identify the best perming model. 

Step 3: Optimize the Model
•    Create a new Google Colab file and name it Optimization_Satisfaction.ipynb
•    Hyperparameter tuning was performed using Grid Search on the XGBoost Classifier, resulting in improved accuracy. 
•    Used feature importance function from XGBoost to identify the most important features. 


Step 4: Write a Report 
Data Exploration:
Heat map Correlation matrix of all features showed Strong Correlations between:
•    Inflight wifi service with Ease of online booking 
•    Cleanliness with inflight entertainment/Food and Drink/Seat Comfort 
•    Baggage handling with inflight service 
•    Flight distance with Business Class 
•    
Correlation matrix of target column “Satisfaction” showed:
Three strongest positive correlations:
•     business class travel 
•    online boarding
•    inflight entertainment
The three strongest negative correlations:
•     personal travel
•    economy Class
•    disloyal customers.

Model Evaluation with Standard Scaling showed: XGBoost Classifier is the best performing model. 
Scaling advantages: 
Enhanced Model Performance 
Improved Convergence Speed 
Equal Contribution of Features: Without standardization, features with larger scales might dominate the learning process, leading to biased results. 
 
Model Evaluation without scaling showed: 
Compared to model performance with scaling technique, four models showed reduced performance while other models were not affected. 
XGBoost Classifier is still best performing model indicating insensitive to scaling. 
Models were found which were sensitive to scaling: 
•    Logistic Regression 
•    K-Nearest Neighbors 
•    Support Vector Machine 
•    Naive Bayes Classifier 

Model Evaluation by Removing Outliers showed:
Used IQR technique to detect and remove outliers. 
Four columns had outliers as follows: 
Number of outliers in Flight Distance: 2847 
Number of outliers in Checkin service: 16059 
Number of outliers in Departure Delay in Minutes: 17970 
Number of outliers in Arrival Delay in Minutes: 17492 
Original dataframe rows: 129880 
Dataframe rows after removing outliers: 82637 
Compared to model performance with scaling techique, all models showed decrease in performance. 
Random Forest is best performing model followed by XGBoost classfier. 
 
Model Evaluation by Log Transformation Technique: 
The difference between 129880 and 82637 is approximately 36.37% of 129880. When we removed outliers, we also lost a chunk of data that impacted our modal perforance. 
Outliers can be data errors or natural values in real world. Considering this dataset, we think outliers in this dataset are values we can get in real world. 
We used Log Transformation Technique: 
To help normalize highly skewed data. 
Mitigate the impact of outliers. This made the dataset more robust to outliers and improve the performance of algorithms.

Compared to model performance with scaling techique, all models showed increase in performance. 
XGBoost Classifier is  best performing model. Although there was minimal difference between accuracy scores (96.40% and 96.37%) between XGBoost Classifier and Random Forest, XGBoost Classifier had an edge over Random Forest in training time speed. 

Model Performance by ROC-AUC: The Area Under the Receiver Operating Characteristic (AUC-ROC) Curve for all models show  XGBoost Classifier is the best performing model. 
 
MODAL OPTIMIZATION:  
Used Grid Search technique for optimization using hyperparameters tuning method. 
 Before Tuning:  
•    Accuracy: 0.9640 
•    The initial XGBoost model, trained with default hyperparameters, achieved an accuracy of approximately 96.40% on the test data. 
 After tuning: 
•    Accuracy: 0.9653 
•    After performing hyperparameter tuning using GridSearchCV, the best XGBoost model achieved an improved accuracy of approximately 96.53% on the test data. 
•    The best parameters found during tuning were: 
 
•    gamma: 0.1 
•    learning_rate: 0.05 
•    max_depth: 9 
•    n_estimators: 200 
 
This improvement in accuracy from 96.40% to 96.53% indicates that the hyperparameter tuning process helped to fine-tune the model's parameters, leading to noticeable enhancement in predictive performance. 
 
Feature Importance: 
Feature Importance Calculation: XGBoost provides built-in functions to calculate feature importance, which helps in understanding the influence of different features on the model's predictions. 
•    Satisfaction of passengers is most strongly influenced by factors such as the online boarding experience, business travel, and the quality of inflight WiFi service.  
•    Features related to the class of travel, customer loyalty, and inflight entertainment also play a notable role.  
•    Other features, although included in the model, contribute significantly less to the prediction outcomes. This insight can guide improvements in service areas that matter most to passengers, potentially leading to higher satisfaction and better model performance. 

Conclusion: 
•    The project successfully identified the most important features influencing customer satisfaction and predicted satisfaction with high accuracy, high precision, and high recall. 
•    The best-performing model after hyperparameter tuning was the XGBoost Classifier, with an accuracy improvement from 96.40% to 96.53%. 
 
•    XGBoost (Extreme Gradient Boosting) is a powerful and popular machine learning algorithm, especially for supervised learning tasks.  
Here are some of the key advantages of using the XGBoost classifier in this project: 
1.     Performance :   XGBoost is known for its high performance in terms of accuracy.  
2.     Speed: Built for speed, perfect for large datasets. 
3.     It can handle unbalanced datasets effectively. 
4.     XGBoost includes regularization parameters (L1 and L2) which help prevent overfitting.  
5.     XGBoost provides built-in functions to calculate feature importance, which helps in understanding the influence of different features on the model's predictions. 
 
 
### Prerequisites
-    Tools need to have installed before project:
•    Google Colab notebook
•    Pandas
•    seaborn
•    Sklearn
•    Matplotlib
•    Numpy
•    Time
•    Xgboost


###References:

https://www.geeksforgeeks.org/auc-roc-curve/                 

https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Roc_curve.svg/220px-Roc_curve.svg.png

https://stackoverflow.com/questions/7370801/how-do-i-measure-elapsed-time-in-python                     

https://medium.com/@Oladapoadebola/airline-passenger-satisfaction-77c47502aa4c                            

https://medium.com/@shivanipickl/what-is-feature-scaling-and-why-does-machine-learning-need-it-104eedebb1c9    
                 
https://www.kaggle.com/discussions/getting-started/159643               

https://juandelacalle.medium.com/best-tips-and-tricks-when-and-why-to-use-logarithmic-transformations-in-statistical-analysis-9f1d72e83cfc          

https://www.geeksforgeeks.org/difference-between-random-forest-vs-xgboost/                            x

https://www.geeksforgeeks.org/auc-roc-curve/                


