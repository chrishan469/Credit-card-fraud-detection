#Credit card fraud detection system using machine learning.
#To address the class imbalance issue and achieve an accuracy of more than 90%
#Perform EDA on the dateset
#Applying different machine learning algorithms
#Training and evaluate our modules on the dataset
#The model is to develop a system where the bank user can fetch transactional records into the system and get a binary non fraud or Fraud answer from the system.
# The evaluation process to justify how well the system can perform for the tested data


#1)Perform EDA on the dateset
#1.1)importing libraries
import numpy as np #for special numerical variables
import pandas as pd  #to read data and for special variables
import seaborn as sns #for special graphical operations
import matplotlib.pyplot as plt #for graphs and plotting things


#1.1)importing dataset
dataset = pd.read_csv(r'C:\Users\User\Downloads\Credit-Card-Fraud-Detection-master(1)\Credit-Card-Fraud-Detection-master\New folder\balanced_dataset.csv')

#1.1)Displaying Dataset Head
print(dataset.head())

#2)Checking For Missing Values In Dataset
print(dataset.isnull().sum())

#to get better view
print(dataset.describe().transpose())


#3)data visualisation
plt.figure(figsize = (6,6))
dataset['Class'].value_counts()
counts = [dataset['Class'].value_counts().iloc[0],dataset['Class'].value_counts().iloc[1]]
labels = ['non fraud Transactions -> ' + str(round((counts[0]/dataset.shape[0]) * 100, 2)) + "%", 'Fraud Transactions -> '  + str(round((counts[1]/dataset.shape[0]) * 100, 2)) + "%"]
plt.pie(counts, labels=labels)
plt.legend()
plt.title('Distribution Of Fraud & non fraud Transactions In Dataset')
plt.show()

# Plotting Correlation Between Different Factors Of Dataset
data_corr = dataset.corr()
fig = plt.figure(figsize = (10,10))
sns.heatmap(data_corr, square=True, cmap='Blues')
plt.title(' Heatmap For the Dataset Variables')
plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,6))
ax1.hist(dataset.Amount[dataset.Class == 1], bins = 30, color='yellow')
ax1.set_title('Fraudalent Transactions')
plt.xlabel('Amount')
ax2.hist(dataset.Amount[dataset.Class == 0], bins = 30, color='blue')
ax2.set_title('non fraud Transactions')
plt.ylabel('Transaction number')
plt.yscale('log')
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,6))
ax1.hist(dataset.Time[dataset.Class == 1], bins = 30, color='yellow')
ax1.set_title('Transactions of Frauds')
plt.xlabel('Time from the Last Transaction')
ax2.hist(dataset.Time[dataset.Class == 0], bins = 30, color='blue')
ax2.set_title('non fraud Transactions')
plt.ylabel('Transaction number')
plt.yscale('log')
plt.show()

###creating blanced dataset
plt.figure(figsize = (6,6))
counts = [dataset['Class'].value_counts().iloc[0],dataset['Class'].value_counts().iloc[1]]
labels = ['Genuine Transactions -> ' + str(round((counts[0]/dataset.shape[0]) * 100, 2)) + "%", 'Fraud Transactions -> '  + str(round((counts[1]/dataset.shape[0]) * 100, 2)) + "%"]
plt.pie(counts, labels=labels)
plt.legend(loc = 'upper right')
plt.title('Distribution Of Genuine & Fraud Transactions In Original Dataset')
plt.show()
dataset['Class'].value_counts()


# Selecting All Entries For Fraudulent Transactions
fraud_trans = dataset[dataset['Class'] == 1]


# Selecting These Entries Randomly using sample() method of Pandas DataFrame
gen_trans = dataset[dataset['Class'] == 0].sample(n = 490, random_state=300)

# Combining Both DataFrames To Form Balanced Dataset
dataset = pd.concat([gen_trans, fraud_trans])

plt.figure(figsize = (6,6))
counts = [dataset['Class'].value_counts().iloc[0],dataset['Class'].value_counts().iloc[1]]
labels = ['Genuine Transactions -> ' + str(round((counts[0]/dataset.shape[0]) * 100, 2)) + "%", 'Fraud Transactions -> '  + str(round((counts[1]/dataset.shape[0]) * 100, 2)) + "%"]
plt.pie(counts, labels=labels)
plt.legend(loc = 'right')
plt.title('Distribution Of Genuine & Fraud Transactions In Balanced Dataset')
plt.show()
dataset['Class'].value_counts()

#Saving Balanced Dataset
dataset.to_csv('balanced_dataset.csv')



##splitting dataset into train and test
# Extracting X from the dataset
X = dataset.iloc[:,:-1]

# Extracting y from the dataset
y = dataset.iloc[:,-1]

# Splitting Dataset Into Train and Test Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 300)


#Checking Sizes Of Train & Test Splits
print('Size of X_train : ' + str(X_train.shape))
print('Size of y_train : ' + str(y_train.shape))
print('Size of X_test : ' + str(X_test.shape))
print('Size of y_test : ' + str(y_test.shape))

##feature scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


##model buildings
##decision tree classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

classifier = DecisionTreeClassifier(random_state=300, max_depth=8, min_samples_split=2)
decision_tree_classifier = classifier
decision_tree_classifier.fit(X_train, y_train)
dt_f1_score = np.mean(cross_val_score(decision_tree_classifier, X_train, y_train, scoring='f1', cv=5))
#####(you can get a idea about f1 scores using this links ===> https://towardsdatascience.com/the-f1-score-bec2bbc38aa6)

print('Decison Tree Classifier F1-Score : ' + str(dt_f1_score))
from sklearn.tree import plot_tree
plt.figure(figsize=(15,10))
plot_tree(decision_tree_classifier)
plt.title('Decision Tree For Classification Of Fraud & Genuine Transaction')
plt.show()

#to save decision Tree Model Using Pickle
import pickle
filename = 'Decison_Tree_Classifier.model'
with open(filename, 'wb') as file:
    pickle.dump(decision_tree_classifier, file)



##Support vector machine classifier
from sklearn.svm import SVC
support_vector_machine_classifier = SVC(kernel='linear', random_state = 300, probability=True)
support_vector_machine_classifier.fit(X_train, y_train)
support_vector_machine_f1_score = np.mean(cross_val_score(support_vector_machine_classifier, X_train, y_train, scoring='f1', cv=5))
print('Support Vector Classifier F1-Score : ' + str(support_vector_machine_f1_score))

# to Save SVM Model Using Pickle
import pickle
filename = 'SVM_Classifier.model'
with open(filename, 'wb') as file:
    pickle.dump(support_vector_machine_classifier, file)

#### Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=300, n_estimators=100, bootstrap=True)
rf_classifier.fit(X_train, y_train)
rf_f1_score = np.mean(cross_val_score(rf_classifier, X_train, y_train, scoring='f1', cv=5))


print('Random Forest Classifier F1-Score : ' + str(rf_f1_score))

# Saving Random Forest Model Using Pickle
import pickle
filename = 'Random_Forest_Classifier.model'
with open(filename, 'wb') as file:
    pickle.dump(rf_classifier, file)

##Logistic Regression
# importing LogisticRgression package
from sklearn.linear_model import LogisticRegression
lg_classifier = LogisticRegression(solver='liblinear')
lg_classifier.fit(X_train, y_train)
lg_f1_score = np.mean(cross_val_score(lg_classifier, X_train, y_train, scoring='f1', cv=5))


print('Logistic Regression F1-Score : ' + str(lg_f1_score))

# Saving Logistic Regression Model Using Pickle
import pickle
filename = 'LGR_Classifier.model'
with open(filename, 'wb') as file:
    pickle.dump(lg_classifier, file)


##to evaluate models

model_eval_df = pd.DataFrame(columns=['Model','Train F1 Score','Test F1 Score','Test Accuracy'])
df_index = 0

##to evaluation decision Tree
predictions = decision_tree_classifier.predict(X_test)

#to Caculate F1 Score
from sklearn.metrics import f1_score, accuracy_score
dt_test_f1_score = f1_score(y_test, predictions)
dt_test_acc_score = accuracy_score(y_test, predictions)

#Adding Scores To DataFrame
model_eval_df.loc[df_index] = ['Decision Tree', dt_f1_score, dt_test_f1_score, dt_test_acc_score]
df_index = df_index + 1

##svm model evaluation
predictions = support_vector_machine_classifier.predict(X_test)

#to Caculate F1 Score
from sklearn.metrics import f1_score, accuracy_score
support_vector_machine_test_f1_score = f1_score(y_test, predictions)
svm_test_acc_score = accuracy_score(y_test, predictions)

# Adding Scores To DataFrame
model_eval_df.loc[df_index] = ['SVM Classifier', support_vector_machine_f1_score,support_vector_machine_test_f1_score, svm_test_acc_score]
df_index = df_index + 1


## Random Forest Evaluation
predictions = rf_classifier.predict(X_test)
# Caculating F1 Score
from sklearn.metrics import f1_score, accuracy_score
rf_test_f1_score = f1_score(y_test, predictions)
rf_test_acc_score = accuracy_score(y_test, predictions)

# Adding Scores To DataFrame
model_eval_df.loc[df_index] = ['Random Forest', rf_f1_score, rf_test_f1_score, rf_test_acc_score]
df_index = df_index + 1

### Logistic Regression Evaluation
predictions = lg_classifier.predict(X_test)

# Caculating F1 Score
from sklearn.metrics import f1_score, accuracy_score
lg_test_f1_score = f1_score(y_test, predictions)
lg_test_acc_score = accuracy_score(y_test, predictions)

# Adding Scores To DataFrame
model_eval_df.loc[df_index] = ['Logistic Regression', lg_f1_score, lg_test_f1_score, lg_test_acc_score]
df_index = df_index + 1


#model performance comparison
print(model_eval_df)


#best model prediction
# Predicting Using SVM Classifier
predictions = support_vector_machine_classifier.predict(X_test)


# Visualising Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
df_cm = pd.DataFrame(cm, index=['Genuine','Fraud'], columns=['Genuine','Fraud'])
sns.heatmap(df_cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Printing Classification Report
from sklearn.metrics import classification_report
print(' Report of classifications')
print(classification_report(y_test, predictions))



