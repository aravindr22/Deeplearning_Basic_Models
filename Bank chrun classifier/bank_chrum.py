import pandas as pd
import matplotlib as pllt
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
import seaborn as sn

df = pd.read_csv('D:\deep learning\datasets\\bank_churn_modelling.csv')
df.head()

def print_unique_values(df):
    for i in df:
        if df[i].dtypes == 'object' or True:
            print(f'{i}: {df[i].unique()}')
            
print_unique_values(df)

df['Gender'].replace({'Male': 1, 'Female': 0}, inplace=True)

drop_columns = ["Surname", "CustomerId", "RowNumber"]
df.drop(drop_columns, axis="columns", inplace=True)

df1 = pd.get_dummies(data=df, columns=["Geography"])
print_unique_values(df1)

col_toScale = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]

scaler = MinMaxScaler()

df1[col_toScale] = scaler.fit_transform(df1[col_toScale])

# =============================================================================
# tenure_churn_no = df[df.Exited == 0].Tenure
# tenure_churn_yes = df[df.Exited == 1].Tenure
#  
# plt.xlabel("Tenure")
# plt.ylabel("No.of Customers")
# plt.hist([tenure_churn_no, tenure_churn_yes], color=["green", "red"], label=["Exited=No","Exited=Yes"])
# plt.legend()
# =============================================================================

X = df1.drop('Exited', axis="columns")
y = df1["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, test_size=0.2)


model = keras.Sequential([
        keras.layers.Dense(20, input_shape=(12,), activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(
        optimizer = "adam",
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

