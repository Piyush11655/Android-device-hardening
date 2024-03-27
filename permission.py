from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn import svm
from sklearn import tree
import pandas as pd

import numpy as np
import seaborn as sns
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
import streamlit as st


df = pd.read_csv('permission dataset.csv.csv',sep=";")

df.columns = map(str.lower, df.columns)

df = df.astype("int64")

df1= df.copy()
df1 = df1.loc[:,df1.columns.str.contains('type')  |  df1.columns.str.contains('write') | df1.columns.str.contains('delete') | df1.columns.str.contains('clear') | df1.columns.str.contains('boot') | df1.columns.str.contains('change')| df1.columns.str.contains('credential')|df1.columns.str.contains('admin')|df1.columns.str.contains('list')|df1.columns.str.contains('secure_storage')|df1.columns.str.contains('notifications')|df1.columns.str.contains('account')|df1.columns.str.contains('destroy')|df1.columns.str.contains('mount')|df1.columns.str.contains('authenticate')|df1.columns.str.contains('privileged')|df1.columns.str.contains('brick')|df1.columns.str.contains('transmit')|df1.columns.str.contains('capture')|df1.columns.str.contains('disable')|df1.columns.str.contains('install')|df1.columns.str.contains('certificate')|df1.columns.str.contains('send')|df1.columns.str.contains('shutdown')|df1.columns.str.contains('start_any_activity')|df1.columns.str.contains('lock')|df1.columns.str.contains('sms')|df1.columns.str.contains('call')|df1.columns.str.contains('danger')|df1.columns.str.contains('voicemail')]

df1 = df1.loc[:, (df1 != 0).any(axis=0)]
X_train, X_test, y_train, y_test = train_test_split(df1.iloc[:, 1:40], df1['type'], test_size=0.20, random_state=40)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
pred = gnb.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print("Naive Bayes")
print("Accuracy: " + str(accuracy))
print(classification_report(pred, y_test, labels=None))

for i in range(3,15,3):
    
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    pred = neigh.predict(X_test)
    accuracy = accuracy_score(pred, y_test)
    print("k-neighbors {}".format(i))
    print("Accuracy: " + str(accuracy))
    print(classification_report(pred, y_test, labels=None))
    print("")
    
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)

y_pred_train_gini = clf_gini.predict(X_train)

clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_en.fit(X_train, y_train)
y_pred_en = clf_en.predict(X_test)

y_pred_train_en = clf_en.predict(X_train)

rdF=RandomForestClassifier(n_estimators=250, max_depth=50,random_state=45)
rdF.fit(X_train,y_train)
pred=rdF.predict(X_test)
cm=confusion_matrix(y_test, pred)

accuracy = accuracy_score(y_test,pred)
def main(binary_string):
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f1f0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Permissions")
    
    # Convert the binary string to a list of integers
    features = [int(bit) for bit in binary_string.split(';')]
   
    # Make prediction using the random forest classifier
    prediction = rdF.predict([features])
    
    # Display the prediction
    if prediction[0] == 0:
        st.write("No virus")
    else:
        st.write("Virus")

if __name__ == "__main__":
    # Read the binary string from a file or use any other method to obtain it
    # binary_string = "0;0;0;0;0;0;1;0;1;0;1;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0"
    # Read the binary string from the file
    with open("binary_permissions.txt", "r") as file:
        binary_string = file.read()

# Now you can use the binary string in your other program

    
    # Run the main function with the binary string as input
    main(binary_string)
