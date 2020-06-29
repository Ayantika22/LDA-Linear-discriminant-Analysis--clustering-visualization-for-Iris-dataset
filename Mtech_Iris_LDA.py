#!/usr/bin/env python
# coding: utf-8

# # LDA  plot for IRIS dataset

# In[1]:


'''Linear Discriminant Analysis or Normal Discriminant Analysis or Discriminant Function Analysis is a dimensionality reduction 
technique which is commonly used for the supervised classification problems.'''

# Importing Datasets From Sklearn

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[2]:


# Loading IRIS Dataset 

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names


# In[3]:


# fitting the LDA model
lda = LDA(n_components=2)
lda_X = lda.fit(X,y).transform(X)


# # LDA Cluster plot

# In[4]:


plt.scatter(lda_X[y == 0, 0], lda_X[y == 0, 1], s =80, c = 'orange', label = 'Iris-setosa')
plt.scatter(lda_X[y == 1, 1], lda_X[y == 1, 0], s =80,  c = 'yellow', label = 'Iris-versicolour')
plt.scatter(lda_X[y == 2, 0], lda_X[y == 2, 1], s =80,  c = 'green', label = 'Iris-virginica')
plt.title('LDA plot for Iris Dataset')
plt.legend()


# In[17]:


'''Assigning colors for graph'''

plt.figure()
colors = ['orange', 'yellow', 'green'] 
lw = 2


# In[18]:


# Plotting the graph for LDA (IRIS dataset)

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(lda_X[y == i, 0], lda_X[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets, preprocessing


# In[4]:


Xtrain, Xtest, y_train, y_test = train_test_split(X, y)
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


# In[5]:



clf = LDA()
clf.fit(Xtrain,y_train)
y_pred=clf.predict(Xtest)

y_pred


# In[31]:


from sklearn.metrics import accuracy_score

print('Accuracy Score:', accuracy_score(y_test, y_pred))
                                   


# In[32]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# In[37]:


# Using knn model for accuracy
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtrain, y_train)
y_pred = knn.predict(Xtest)
y_pred


# In[35]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('Confusion matrix \n',  confusion_matrix(y_test, y_pred))
print('Classification \n', classification_report(y_test, y_pred))


# In[36]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# In[6]:


Xtrain, Xtest, y_train, y_test = train_test_split(X, y)


# # Logistic Regression Accuracy 

# In[7]:


#Logistic Regression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Logistic Regression :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for LR

# In[8]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # K-Nearest Neighbors Accuracy

# In[9]:


#K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("K Nearest Neighbors :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for KNN

# In[10]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Support Vector Machine Accuracy

# In[11]:


#Support Vector Machine
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Support Vector Machine:")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for SVM

# In[12]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Gaussian Naive Bayes Accuracy

# In[13]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Gaussian Naive Bayes :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for GNB

# In[14]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Decision Tree Classifier Accuracy

# In[15]:


#Decision Tree Classifier
from sklearn.model_selection import train_test_split
Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

classifier = DT(criterion='entropy', random_state=0)
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
print("Decision Tree Classifier :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for DTC

# In[16]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Random Forest Classifier Accuracy

# In[19]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier as RF
classifier = RF(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
print("Random Forest Classifier :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for RFC

# In[20]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# In[ ]:




