from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from chart_regression import chart_regression
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np

data = pd.read_pickle('train_data.pkl')

print(data)
# Import data to use for regression
X = data[['B_X', 'B_Y', 'B_Z', 'EDP_X', 'EDP_Y', 'EDP_Z']]
y = data['IsEdi']

# Import Edi Data to plot
#df_edi = data[['EDI_X', 'EDI_Y', 'EDI_Z']].copy()

# Split data into test and train data
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)

# Create the regression
reg = linear_model.LinearRegression()

# Train the regression
reg.fit(train_x, train_y)
print(reg.coef_, reg.intercept_)

# Using regression to predict
pred_y = reg.predict(test_x)

# Normalize data for prediction matrix
threshold = 0.9

# TODO: Make more efficient
norm_pred_y = []
for x in pred_y: # lambda x: 1 if np.abs(x) >= threshold else 0
    if np.abs(x) >= threshold:
        norm_pred_y.append(1)
    else:
        norm_pred_y.append(0)

# Create confusion matrix
c_matrix = confusion_matrix(test_y, norm_pred_y)

# Display confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(c_matrix, cmap='Blues', vmin=-300, vmax=300)

# Plot color bar
plt.colorbar(im)

# Format ticks
ax.set_xticks(np.arange(c_matrix.shape[0]))
ax.set_yticks(np.arange(c_matrix.shape[1]))

labels = np.array([['True Neg', 'False Pos'], ['False Neg', 'True Pos']])

# Loop over data dimensions and create text annotations.
for i in range(c_matrix.shape[1]):
    for j in range(c_matrix.shape[0]):
        output = str(labels[i, j]) + '\n' + str(c_matrix[i, j])
        text = ax.text(j, i, output,
                       ha="center", va="center", color="w")


ax.set_title("Confusion Matrix Heat Map for Linear Regression")
fig.tight_layout()

plt.show()

# Plot on XY Plane
chart_regression(test_x, test_y, norm_pred_y, 'Linear')

print("Complete!")
