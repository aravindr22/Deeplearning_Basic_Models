import pandas as pd
import matplotlib as pllt
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
import seaborn as sn

df = pd.read_csv('D:\deep learning\Chrum classifier\customer_chrun.csv')

df.drop('customerID', axis='columns', inplace=True)

#Converting string to integer in totalcaharges column
pd.to_numeric(df.TotalCharges, errors="coerce")

#dropping the rows which has null in total charnges

df1 = df[df.TotalCharges != ' ']

df1.TotalCharges = pd.to_numeric(df1.TotalCharges)

# =============================================================================
# tenure_churn_no = df1[df1.Churn == 'No'].tenure
# tenure_churn_yes = df1[df1.Churn == 'Yes'].tenure
# 
# plt.xlabel("Tenure")
# plt.ylabel("No.of Customers")
# plt.hist([tenure_churn_no, tenure_churn_yes], color=["green", "red"], label=["Churn=Yes","Churn=No"])
# plt.legend()
# 
# tenure_churn_no = df1[df1.Churn == 'No'].MonthlyCharges
# tenure_churn_yes = df1[df1.Churn == 'Yes'].MonthlyCharges
# 
# plt.xlabel("Monthly Charges")
# plt.ylabel("No.of Customers")
# plt.hist([tenure_churn_no, tenure_churn_yes], color=["green", "red"], label=["Churn=Yes","Churn=No"])
# plt.legend()
# =============================================================================

def print_unique_values(df):
    for i in df:
        if df[i].dtypes == 'object' or True:
            print(f'{i}: {df[i].unique()}')

df1.replace("No internet service", "No", inplace=True)
df1.replace("No phone service", "No", inplace=True)
print_unique_values(df1)
df1.head()

col_Names = ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity",
             "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
             "StreamingMovies", "PaperlessBilling", "Churn"]

for col in col_Names:
    df1[col].replace({"Yes": 1, "No": 0}, inplace=True)
    
df1["gender"].replace({"Male": 0, "Female": 1}, inplace=True)

#1-hot encoding for columns multiple unique values

df2 = pd.get_dummies(data = df1, columns=["PaymentMethod", "InternetService", "Contract"])
df2.columns
print_unique_values(df2)

cols_toScale = ["tenure", "MonthlyCharges", "TotalCharges"]

scaler = MinMaxScaler()

df2[cols_toScale] = scaler.fit_transform(df2[cols_toScale])
df2.dtypes

X = df2.drop("Churn", axis="columns")
y = df2["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)


model = keras.Sequential([
        keras.layers.Dense(20, input_shape=(26,), activation="relu"),
        #keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

model.fit(X_train, y_train, epochs=100)

model.evaluate(X_test, y_test)

yp = model.predict(X_test)
yp = yp > 0.5
yp = yp.astype(int)

print(classification_report(y_test, yp))

#confusion matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=yp)

plt.figure(figsize=(5,3))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
