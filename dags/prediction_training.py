import joblib
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from collecting_data import *
from sklearn import preprocessing

# Call the function to get X_np, y, and df
X_np, y, df = data_sensor()


def prediction_of_model():
    print("iske andar aya brooo")
    # scaler = MinMaxScaler()
    # normalized_features = scaler.fit_transform(X_np)
    model = joblib.load('data/1st_iteration_NN_model_with_normalizedData.pkl')

    # model_path  ='1st_iteration_NN_model.pkl'

    # Create a directory with today's date
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_directory = os.path.join('.', today_date)

    # Check if the directory exists, and create it if not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    predicted_normalized_features = model.predict(X_np)
    predicted_label_full_dataset = [
        np.argmax(i) for i in predicted_normalized_features]

    print("evaluating model")

    print(model.evaluate(X_np, y))

    cm = tf.math.confusion_matrix(
        labels=y, predictions=predicted_label_full_dataset)
    
    le = preprocessing.LabelEncoder()
    le.fit(df.classification)
    df['categorical_label'] = le.transform(df.classification)
    df['categorical_label']
    original_labels = le.inverse_transform(np.unique(y))

    
    # saving the first heatmap
    import seaborn as sn
    plt.figure(figsize=(9, 7))
    sn.heatmap(cm, annot=True, fmt='d', xticklabels=original_labels,
               yticklabels=original_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig(os.path.join(output_directory, 'heatmap1.png'))

    
    
    sum_rows = np.sum(cm, axis=1)
    cm1 = tf.convert_to_tensor(
        np.array([cm[i]/sum_rows[i] for i in range(len(cm))]))
    # cm1 = cm[0]/sum_rows[0]
    
    # second heatmap shows us the percentage of accuracy of the predicted values
    cm2 = cm1*100

    plt.figure(figsize=(9, 7))
    sn.heatmap(cm2, annot=True, fmt='.2f',
               xticklabels=original_labels, yticklabels=original_labels)
    plt.title('Predicted Percentages')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    
    plt.savefig(os.path.join(output_directory, 'heatmap2.png'))
    

    diagonal_entries_of_predicted_percentages = np.diagonal(cm2)

    predicted_percentages = {key: value for key, value in zip(
        original_labels, diagonal_entries_of_predicted_percentages)}
    print(predicted_percentages)


def main():
    prediction_of_model()