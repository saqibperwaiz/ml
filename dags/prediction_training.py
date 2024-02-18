import joblib
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import tensorflow as tf
from training_model import model, normalized_features, y, le
import matplotlib.pyplot as plt
import pandas as pd


def prediction_of_model():
    model = joblib.load('data/1st_iteration_NN_model_with_normalizedData.pkl')

    # model_path  ='1st_iteration_NN_model.pkl'



    predicted_normalized_features  = model.predict(normalized_features)
    predicted_label_full_dataset = [np.argmax(i) for i in predicted_normalized_features]

    print("evaluating model")

    print(model.evaluate(normalized_features,y))

    cm = tf.math.confusion_matrix(labels = y,predictions = predicted_label_full_dataset)

    print(cm)


    original_labels = le.inverse_transform(np.unique(y))



    import seaborn as sn
    plt.figure(figsize = (9,7))
    sn.heatmap(cm,annot = True, fmt = 'd',xticklabels=original_labels, yticklabels=original_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')



    sum_rows  = np.sum(cm,axis=1)
    cm1 = tf.convert_to_tensor(np.array([cm[i]/sum_rows[i] for i in range(len(cm))]))
    # cm1 = cm[0]/sum_rows[0]

    cm2 = cm1*100


    plt.figure(figsize = (9,7))
    sn.heatmap(cm2,annot = True, fmt = '.2f',xticklabels=original_labels, yticklabels=original_labels)
    plt.title('Predicted Percentages')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    diagonal_entries_of_predicted_percentages = np.diagonal(cm2)

    predicted_percentages = {key:value for key,value in zip(original_labels,diagonal_entries_of_predicted_percentages)}
    print(predicted_percentages)