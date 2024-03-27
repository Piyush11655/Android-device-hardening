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

import subprocess




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







# Load the pre-trained model


def run_permission_script(app_name):
    output = ""
    if app_name.lower() in ["whatsapp", "snapchat", "dream11"]:
        # Run Permissio.py with the given app_name
        command = ["python", "Permissio.py", app_name]
        result = subprocess.run(command,input=app_name, capture_output=True, text=True)
        output = result.stdout
    else:
        # Run p.py with the given app_name
        command = ["python", "p.py", app_name]
        result = subprocess.run(command,input=app_name, capture_output=True, text=True)
        output = result.stdout

    return output

def main():
    st.title("App Permissions and Virus Detection")
    
    # Get the app name from the user
    app_name = st.text_input("Enter the name of the app ").strip()
    
    if app_name:
        # Get permissions for the specified app name
        permissions_output = run_permission_script(app_name)
        
        # Display permissions
        st.subheader("Permissions for the App:")
        st.text_area("Permissions", value=permissions_output, height=300)
        
        # Button to check for virus
        if st.button("Check for Virus"):
            permissions = [
        "android.permission.READ_SMS",
        "android.permission.MANAGE_EXTERNAL_STORAGE",
        "android.permission.ACCESS_BLUETOOTH_SHARE",
        "android.permission.ACCESS_DRM_CERTIFICATES",
        "android.permission.ACCESS_FINE_LOCATION",
        "android.permission.POST_NOTIFICATION",
        "android.permission.ACCESS_CONTENT_PROVIDERS_EXTERNALLY",
        "android.permission.ACCESS_DOWNLOAD_MANAGER",
        "android.intent.category.MASTER_CLEAR.permission.C2D_MESSAGE",
        "android.os.cts.permission.TEST_GRANTED",
        "android.permission.ACCESS_INPUT_FLINGER",
        "android.permission.ACCESS_KEYGUARD_SECURE_STORAGE",
        "android.permission.ACCESS_BLUETOOTH_SHARE",
        "android.permission.ACCESS_CACHE_FILESYSTEM",
        "android.permission.READ_CALL_LOGS",
        "android.permission.CAMERA",
        "android.permission.ACCESS_CONTENT_PROVIDERS_EXTERNALLY",
        "android.permission.ACCESS_DOWNLOAD_MANAGER",
        "android.permission.ACCESS_DOWNLOAD_MANAGER_ADVANCED",
        "android.permission.ACCESS_DRM_CERTIFICATES",
        "android.permission.ACCESS_WIFI_STATE",
        "android.permission.RECORD_AUDIO",
        "android.permission.ACCESS_CHECKIN_PROPERTIES",
        "android.permission.ACCESS_COARSE_LOCATION",
        "android.permission.ACCESS_CONTENT_PROVIDERS_EXTERNALLY",
        "android.permission.ACCESS_DOWNLOAD_MANAGER",
        "android.permission.ACCESS_DOWNLOAD_MANAGER_ADVANCED",
        "android.permission.ACCESS_DRM_CERTIFICATES",
        "android.intent.category.MASTER_CLEAR.permission.C2D_MESSAGE",
        "android.os.cts.permission.TEST_GRANTED",
        "android.permission.AUTHENTICATE_ACCOUNTS",
        "android.permission.BACKUP",
        "android.permission.BATTERY_STATS",
        "android.permission.ACCESS_CACHE_FILESYSTEM",
        "android.permission.ACCESS_CHECKIN_PROPERTIES",
        "android.permission.ACCESS_CONTENT_PROVIDERS_EXTERNALLY",
        "android.permission.ACCESS_COARSE_LOCATION",
        "android.permission.INTERNET",
        "android.permission.WAKE_LOCK",
    ]

    # Create a list of 0's and 1's
            binary_list = ['1' if permission in permissions_output else '0' for permission in permissions]

    # Join the list into a comma-separated string
            binary_string = ';'.join(binary_list)
            st.write(binary_string)
            

            # Process permissions to binary string
        
            
            
            # Predict virus using the model
            #binary_string = "0;0;0;0;0;0;1;0;1;0;1;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0"
            features = [int(bit) for bit in binary_string.split(';')]
   
    # Make prediction using the random forest classifier
            prediction = rdF.predict([features])
    

            
            # Display prediction
            st.subheader("Virus Prediction:")
            if prediction[0] == 0:
                st.write("No virus detected.")
            else:
                st.write("Virus detected!")
    

if __name__ == "__main__":
    main()
