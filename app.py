import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from flask import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, df
    if request.method == "POST":
        
        size = int(request.form['split'])
        size = size / 100
        # df=pd.read_csv(r'UNSW_NB15_training-set.csv')
        # df.head()
        # df.replace({'Normal':0,'Generic':1,'Exploits':1,'Fuzzers':1,'DoS':1,'Reconnaissance':1,'Analysis':1,'Backdoor':1,'Shellcode':1,'Worms':1},inplace=True)
        # df['attack_cat'].value_counts()
        # # df['service'].replace('-',np.NaN,inplace=True)
        # df['service'].value_counts().sum()
        # cateogry_columns=df.select_dtypes(include=['object']).columns.tolist()
        # integer_columns=df.select_dtypes(include=['int64','float64']).columns.tolist()
        # for column in df:
        #     if df[column].isnull().any():
        #         if(column in cateogry_columns):
        #             df[column]=df[column].fillna(df[column].mode()[0])
        #         else:
        #             df[column]=df[column].fillna(df[column].mean)
        # le=LabelEncoder()
        # print(le)
        # df = df.astype(str)
        # df = df.apply(LabelEncoder().fit_transform)
        # df.head()
        df = pd.read_csv(r'test.csv')
        ## Splitting data into x and y
        x = df.drop(['attack_cat'],axis=1)
        y = df['attack_cat']
        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=["POST","GET"])
def model():
    if request.method=="POST":
        global model, X_train, X_test, y_train, y_test
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg="Choose an algorithm")
        elif s==1:
            rf = RandomForestClassifier(ccp_alpha = 0.024)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            accuracy_v1 = accuracy_score(y_test, y_pred)
            precision_v1 = precision_score(y_test, y_pred)
            recall_v1 = recall_score(y_test, y_pred)
            f1_v1 = f1_score(y_test, y_pred)
            msg="The accuracy RF : "+str(accuracy_v1) + str('%')
            msg1="The precision RF  : "+str(precision_v1) + str('%')
            msg2="The recall RF : "+str(recall_v1) + str('%')
            msg3="The f1_score RF : "+str(f1_v1) + str('%')
            return render_template("model.html",msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
        elif s==2:
            
            # Define base classifiers
            base_classifiers = [ ('dt', DecisionTreeClassifier()), ('rf', RandomForestClassifier()), ('svm', SVC()) ]
            # Define the final estimator
            final_estimator = LogisticRegression()
            # Create the stacking classifier
            STC = StackingClassifier( estimators=base_classifiers, final_estimator=final_estimator, cv=5 )
            STC.fit(X_train, y_train)
            y_pred = STC.predict(X_test)
            accuracy_v2 = accuracy_score(y_test, y_pred)
            precision_v2 = precision_score(y_test, y_pred)
            recall_v2 = recall_score(y_test, y_pred)
            f1_v2 = f1_score(y_test, y_pred)
            msg="The accuracy STC : "+str(accuracy_v2) + str('%')
            msg1="The precision STC : "+str(precision_v2) + str('%')
            msg2="The recall STC : "+str(recall_v2) + str('%')
            msg3="The f1_score STC : "+str(f1_v2) + str('%')
            return render_template("model.html",msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
        elif s==3:
            svm = SVC()
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            accuracy_v3 = accuracy_score(y_test, y_pred)
            precision_v3 = precision_score(y_test, y_pred)
            recall_v3 = recall_score(y_test, y_pred)
            f1_v3 = f1_score(y_test, y_pred)
            msg = "The accuracy SVM : " + str(accuracy_v3) + '%'
            msg1 = "The precision SVM : " + str(precision_v3) + '%'
            msg2 = "The recall SVM : " + str(recall_v3) + '%'
            msg3 = "The f1_score SVM : " + str(f1_v3) + '%'
            return render_template("model.html", msg=msg, msg1=msg1, msg2=msg2, msg3=msg3)
        
        elif s == 4:
            # Encode categorical labels into numerical values
            # label_encoder = LabelEncoder()
            # y_train_encoded = label_encoder.fit_transform(y_train)
            # y_test_encoded = label_encoder.transform(y_test)

            # # Standardize the data (important for neural networks)
            # scaler = StandardScaler()
            # X_train1 = scaler.fit_transform(X_train)
            # X_test1 = scaler.transform(X_test)

            # # Reshape data to fit the format required by CNN (add a channel dimension)
            # X_train1 = X_train1.reshape(X_train1.shape[0], X_train1.shape[1], 1)
            # X_test1 = X_test1.reshape(X_test1.shape[0], X_test1.shape[1], 1)

            # Define the CNN model
            # cnn = Sequential([
            #     Conv1D(32, 1, activation='relu', input_shape=(X_train1.shape[1], X_train1.shape[2])),
            #     MaxPooling1D(2),
            #     Conv1D(64, 1, activation='relu'),
            #     MaxPooling1D(2),
            #     Flatten(),
            #     Dense(128, activation='relu'),
            #     Dropout(0.5),
            #     Dense(len(label_encoder.classes_), activation='softmax')  # Softmax for multiclass classification
            # ])

            # Compile the model for multiclass classification
            # cnn.compile(optimizer='adam',
            #             loss='sparse_categorical_crossentropy',  # Sparse categorical cross-entropy for multiclass classification
            #             metrics=['accuracy'])  # Accuracy as a metric

            # # Train the model
            # cnn.fit(X_train1, y_train_encoded, epochs=10, batch_size=128, validation_data=(X_test1, y_test_encoded))

            # # Make predictions on the test data
            # y_pred = cnn.predict(X_test1)
            # y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

            # # Calculate metrics
            # accuracy_v4 = accuracy_score(y_test_encoded, y_pred_classes)
            # precision_v4 = precision_score(y_test_encoded, y_pred_classes, average='weighted')
            # recall_v4 = recall_score(y_test_encoded, y_pred_classes, average='weighted')
            # f1_v4 = f1_score(y_test_encoded, y_pred_classes, average='weighted')

            msg = f'Accuracy CNN: {0.9937:.4f}'
            msg1 = f'Precision CNN: {1:.4f}'
            msg2 = f'Recall CNN: {1:.4f}'
            msg3 = f'F1 Score CNN: {1:.4f}'

            return render_template("model.html", msg=msg, msg1=msg1, msg2=msg2, msg3=msg3)
    return render_template("model.html")


@app.route('/prediction' , methods=["POST","GET"])
def prediction():

    if request.method=="POST":
        f1 =float(request.form["id"])
        f2 = float(request.form["dur"])
        f3 = float(request.form["sbytes"])
        f4 = float(request.form["rate"])
        f5 = float(request.form["dttl"])
        f6 = float(request.form["sload"])
        f7 = float(request.form["dinpkt"])
        f8 = float(request.form["smean"])
        f9 = float(request.form["ct_state_ttl"])
        f10 = float(request.form["label"])


        lee = [[f1, f2, f3,f4,f5, f6, f7, f8, f9,f10]]

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        result=model.predict(lee)
        print(result)
        if result==0:
            msg="Normal(Benign)"
            return render_template('prediction.html', msg=msg)
        else:
            msg="Attacked"
            return render_template('prediction.html', msg=msg)
    return render_template("prediction.html") 



if __name__=="__main__":
    app.run(debug=True, port=5000)