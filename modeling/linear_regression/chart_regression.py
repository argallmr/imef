import pandas as pd
import matplotlib.pyplot as plt

"""
chart_regression.py - contains code that plots the confusion matrix on the edi xy plane.
"""


def chart_regression(test_x, test_y, norm_pred_y, type):

    df_tp = pd.DataFrame()
    df_tn = pd.DataFrame()
    df_fp = pd.DataFrame()
    df_fn = pd.DataFrame()

    # Organizing results 'confusion matrix' comparing prediction with actual results
    for i in range(0, len(test_y)):
        if test_y.iloc[i] == norm_pred_y[i] and test_y.iloc[i] == 1:
            #print("True Positive")
            df_tp = df_tp.append(test_x.iloc[i])
        elif test_y.iloc[i] == norm_pred_y[i] and test_y.iloc[i] == 0:
            #print("True Negative")
            df_tn = df_tn.append(test_x.iloc[i])
        elif test_y.iloc[i] != norm_pred_y[i] and test_y.iloc[i] == 1:
            #print("False Positive")
            df_fp = df_fp.append(test_x.iloc[i])
        elif test_y.iloc[i] != norm_pred_y[i] and test_y.iloc[i] == 0:
            #print("False Negative")
            df_fn = df_fn.append(test_x.iloc[i])
        else:
            print('Invalid Data Row')

        if (i % 1000 == 0):
            print('Progress...', i/len(test_y), '%')


    # Graphing results
    fig, ax = plt.subplots()

    ax.scatter(df_tp['EDP_X'], df_tp['EDP_Y'], s=2, c='g', label="True Positive")
    ax.scatter(df_tn['EDP_X'], df_tn['EDP_Y'], s=2, c='b', label="True Negative")
    ax.scatter(df_fp['EDP_X'], df_fp['EDP_Y'], s=2, c='r', label="False Positive")
    ax.scatter(df_fn['EDP_X'], df_fn['EDP_Y'], s=2, c='m', label="False Negative")

    ax.set_xlabel('EDP X')
    ax.set_ylabel('EDP Y')
    ax.set_title('MMS Data through ' + type + ' Regression to Detect EDI Data Points')

    ax.legend()
    ax.grid(True)

    plt.show()