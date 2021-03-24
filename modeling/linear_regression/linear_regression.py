from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np

#data = pd.read_pickle('train_data.pkl')
data = pd.read_csv('dummy_data.csv')

# Index of half of the data
half = int(len(data['B_X']) / 2)

train_x = data[['B_X', 'B_Y', 'B_Z', 'EDP_X', 'EDP_Y', 'EDP_Z']].iloc[:half]
predict_x = data[['B_X', 'B_Y', 'B_Z', 'EDP_X', 'EDP_Y', 'EDP_Z']].iloc[half:]

train_y = data['IsEdi'].iloc[:half]
predict_y = data['IsEdi'].iloc[half:]

# Create the regression
reg = linear_model.LinearRegression()

# Train the regression
reg.fit(train_x, train_y)
print(reg.coef_, reg.intercept_)

# Using regression to predict
prediction = reg.predict(predict_x)

# Creating dataframe to compare
df = data[['EDI_X', 'EDI_Y', 'EDI_Z', 'IsEdi']].iloc[half:]
df['Certainty'] = prediction

# Normalize prediction to be zero or one
threshold = .9
df['Predicted'] = df['Certainty'].apply(lambda x: 1 if np.abs(x) >= threshold else 0)

df_tp = pd.DataFrame()
df_tn = pd.DataFrame()
df_fp = pd.DataFrame()
df_fn = pd.DataFrame()

# Organizing results 'confusion matrix'
for i in range(0, len(df)):
    if df['IsEdi'].iloc[i] == df['Predicted'].iloc[i] and df['Predicted'].iloc[i] == 1:
        # print("True Positive")
        df_tp = df_tp.append(df.iloc[i])
    elif df['IsEdi'].iloc[i] == df['Predicted'].iloc[i] and df['Predicted'].iloc[i] == 0:
        # print("True Negative")
        df_tn = df_tn.append(df.iloc[i])
    elif df['IsEdi'].iloc[i] != df['Predicted'].iloc[i] and df['Predicted'].iloc[i] == 1:
        # print("False Positive")
        df_fp = df_fp.append(df.iloc[i])
    elif df['IsEdi'].iloc[i] != df['Predicted'].iloc[i] and df['Predicted'].iloc[i] == 0:
        # print("False Negative")
        df_fn = df_fn.append(df.iloc[i])
    else:
        print('Invalid Data Row')

# Graphing results
fig, ax = plt.subplots()

ax.scatter(df_tp['EDI_X'], df_tp['EDI_Y'], s=2, c='g', label="True Positive")
ax.scatter(df_tn['EDI_X'], df_tn['EDI_Y'], s=2, c='b', label="True Negative")
# ax.scatter(df_fp['EDI_X'], df_fp['EDI_Y'], s=2, c='r', label="False Positive")
ax.scatter(df_fn['EDI_X'], df_fn['EDI_Y'], s=2, c='m', label="False Negative")

ax.set_xlabel('EDI X')
ax.set_ylabel('EDI Y')
ax.set_title('MMS Data through Linear Regression to Detect EDI Data Points')

ax.legend()
ax.grid(True)

plt.show()
