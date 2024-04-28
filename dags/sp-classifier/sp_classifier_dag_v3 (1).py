# DAG exhibiting task flow paradigm in airflow 2.0
# https://airflow.apache.org/docs/apache-airflow/2.0.2/tutorial_taskflow_api.html
# Modified for our use case

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
from datetime import timedelta
import io
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.S3_hook import S3Hook
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.hooks.subprocess import SubprocessHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

import sys
import subprocess

TEST_BUCKET = 'test-airflow-s3-bucket'
ENV_BUCKET = 'sp-classifier-mwaa'
default_args = {
    'owner': 'airflow',
    'retry': 5,
    'retry_delay': timedelta(minutes=1),
    # "email": "ahmadareeb3026@gmail.com",
    # "email_on_failure": True,
    # "email_on_retry": False,

}


@dag(default_args=default_args, schedule_interval="@monthly", start_date=days_ago(2), tags=['SP_Classifier'])
def sp_classifier_dag_v4_using_taskflow_api():

    # @task
    # def installing_dependencies_using_subprocess():

    #     subprocess.check_call(
    #         [sys.executable, "-m", "pip", "install", "numpy"])
    #     subprocess.check_call(
    #         [sys.executable, "-m", "pip", "install", "pandas"])
    #     subprocess.check_call(
    #         [sys.executable, "-m", "pip", "install", "seaborn"])
    #     subprocess.check_call(
    #         [sys.executable, "-m", "pip", "install", "scikit-learn"])
    #     subprocess.check_call(
    #         [sys.executable, "-m", "pip", "install", "tensorflow"])
    #     subprocess.check_call(
    #         [sys.executable, "-m", "pip", "install", "joblib"])
    #     subprocess.check_call(
    #         [sys.executable, "-m", "pip", "install", "matplotlib"])
    #     subprocess.check_call(
    #         [sys.executable, "-m", "pip", "install", "s3fs"])

    #     return {"message": "Dependencies installed successfully"}

    @task
    def printCurrent_directory():
        import os
        print(os.listdir())

    @task()
    def pre_processing_data():
        import os
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        import joblib
        from sklearn import preprocessing
        from sklearn.preprocessing import normalize
        import pickle
        from io import StringIO
        from sklearn.preprocessing import MinMaxScaler

        hook = S3Hook()
        # file_content = hook.read_key(
        #     key="data/dataset_after_training.csv", bucket_name=ENV_BUCKET
        # )
        # csv_string_io = StringIO(file_content.decode('utf-8'))
        # df = pd.read_csv(csv_string_io)

        s3_object = hook.get_key(
            key="data/dataset_after_training.csv", bucket_name=ENV_BUCKET)

        # Read the CSV file contents
        if s3_object:
            csv_bytes = s3_object.get()['Body'].read()
            df = pd.read_csv(io.BytesIO(csv_bytes))
            # Now df contains your CSV data as a DataFrame
        else:
            print("S3 key not found")

        X = df[['previous_trend_duration', 'previous_percentage', 'harpoon_output_signal', 'harpoon_fuel', 'harpoon_burnout', 'harpoon_h9', 'harpoon_h21', 'harpoon_h14', 'top_gun_output_signal', 'top_gun_top_gun_a', 'top_gun_top_gun_b', 'top_gun_fuel', 'top_gun_burnout', 'maverick_output_signal', 'maverick_fuel', 'maverick_burnout', 'wind_output_signal', 'wind_crossover', 'sonar_output_signal', 'sonar_fuel', 'sonar_burnout', 'Storm', 'Storm_Actual', 'Storm_Impact', 'Storm_MA30', 'Storm_MA7', 'binaryverdict_grouchy', 'BinaryVerdictBernadotte',
                'binaryverdict_Marmont', 'bernadotte_random_forest',
                'Marmont_random_forest', 'Grouchy_random_forest']]
        y = df['classification']

        X_np = np.array(X)
        column_means = np.nanmean(X_np, axis=0)
        nan_indices = np.isnan(X_np)
        X_np[nan_indices] = np.take(column_means, np.where(nan_indices)[1])

        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(X_np)

        encoded_scaler_label = preprocessing.LabelEncoder()
        encoded_scaler_label.fit(df.classification)
        df['categorical_label'] = encoded_scaler_label.transform(
            df.classification)
        df['categorical_label']

        categorical_labels = df.categorical_label

        X_train, X_test, y_train, y_test = train_test_split(
            normalized_features, categorical_labels, test_size=0.3)

        np.count_nonzero(np.isnan(normalized_features))

        # y_test.shape

        X_train_norm = normalize(X_train)

        my_dict = {
            'X_train': X_train,
            'y_train': y_train,  # Convert Series to list
            'X_test': X_test,
            'y_test': y_test,    # Convert Series to list
            'encoded_scaler_label': encoded_scaler_label,
            'categorical_labels': categorical_labels,  # Convert Series to list
            'normalized_features': normalized_features
        }

        # current_datetime = datetime.now().strftime("%Y%m%d")
        # folder_path = f"/data/{current_datetime}/"

        # os.makedirs(folder_path, exist_ok=True)

        # Save pickled data to a file within the folder

        pickled_data = pickle.dumps(my_dict)
        key_of_preprocessing_output = 'preprocessing output/preprocessing_ouput.pkl'

        hook = S3Hook()
        print(hook.check_for_bucket(bucket_name=ENV_BUCKET))
        hook.load_bytes(
            bytes_data=pickled_data,
            key=key_of_preprocessing_output,
            bucket_name=ENV_BUCKET,
            replace=True)

    # Return the file path
        return {"key of preprocessing output": key_of_preprocessing_output}

    @task(multiple_outputs=True)
    def training(json_key_of_preprocessing_output):

        import tensorflow as tf
        from tensorflow import keras
        # import matplotlib.pyplot as plt
        import pandas as pd
        import pickle
        # %matplotlib inline
        import numpy as np
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import normalize
        from sklearn.metrics import roc_curve, roc_auc_score
        import joblib

        key_of_preprocessing_output = json_key_of_preprocessing_output[
            'key of preprocessing output']

        hook = S3Hook()
        s3_object = hook.get_key(
            key=key_of_preprocessing_output, bucket_name=ENV_BUCKET
        )

        # Read the contents of the S3 object
        file_content_of_preprocessed = s3_object.get()["Body"].read()

        # Unpickle the data
        unpickled_data = pickle.loads(file_content_of_preprocessed)

        X_train = unpickled_data['X_train']
        y_train = unpickled_data['y_train']

        model = keras.Sequential([
            # Dense means all the neurons in 1 st layer connect with all the neuron in 2nd layer
            keras.layers.Dense(50, input_shape=(32,), activation='relu'),
            keras.layers.Dense(8, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            # sparce means out output is an integer that is basically train_y
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(X_train, y_train, epochs=30)
        # joblib.dump(model, 'data/1st_iteration_NN_model_with_normalizedData.pkl')

        key_of_training_output = "trained model/trained_model.pkl"

        pickled_data_train = pickle.dumps(model)

        print(hook.check_for_bucket(bucket_name=ENV_BUCKET))
        hook.load_bytes(
            bytes_data=pickled_data_train,
            key=key_of_training_output,
            bucket_name=ENV_BUCKET,
            replace=True)

        return {"key of trained model": key_of_training_output}

    @task()
    def prediction_and_analyzing(json_key_of_preprocessed_data, json_key_of_trained_model):

        import pickle
        import numpy as np
        import seaborn as sn
        import matplotlib.pyplot as plt
        import tensorflow as tf

        hook = S3Hook()
        key_of_preprocessing_output = json_key_of_preprocessed_data['key of preprocessing output']

        hook = S3Hook()
        s3_object = hook.get_key(
            key=key_of_preprocessing_output, bucket_name=ENV_BUCKET
        )

        # Read the contents of the S3 object
        file_content_of_preprocessed = s3_object.get()["Body"].read()

        # Unpickle the data
        unpickled_data = pickle.loads(file_content_of_preprocessed)

        normalized_features = unpickled_data['normalized_features']
        y_test = unpickled_data['y_test']
        encoded_scaler_label = unpickled_data['encoded_scaler_label']
        categorical_labels = unpickled_data['categorical_labels']

        # import the model from s3 trained model folder
        training_model_key = json_key_of_trained_model['key of trained model']

        s3_object_trained_model = hook.get_key(
            key=training_model_key, bucket_name=ENV_BUCKET
        )
        trained_model_pickle = s3_object_trained_model.get()["Body"].read()

        # Load the model by un pickling it
        model = pickle.loads(trained_model_pickle)

        # predicting the labels on whole data set
        predicted_normalized_features = model.predict(normalized_features)

        # extracting the labels with greater percentage predicted
        predicted_label_full_dataset = [
            np.argmax(i) for i in predicted_normalized_features]

        # evaluating the model
        model_evaluation = model.evaluate(
            normalized_features, categorical_labels)

        print(model_evaluation)

        # confusion matrix

        cm = tf.math.confusion_matrix(
            labels=categorical_labels, predictions=predicted_label_full_dataset)

        # converting the decoded labels to original labels
        original_labels = encoded_scaler_label.inverse_transform(
            np.unique(categorical_labels))

        # plotting the confusion matrix using seaborn

        plt.figure(figsize=(9, 7))
        sn.heatmap(cm, annot=True, fmt='d',
                   xticklabels=original_labels, yticklabels=original_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')

        # saving the confusion matrix
        plt.savefig('confusion_matrix.png')

        # calculating the percentage recall of predicted labels

        sum_rows = np.sum(cm, axis=1)
        cm1 = tf.convert_to_tensor(
            np.array([cm[i]/sum_rows[i] for i in range(len(cm))]))
        # cm1 = cm[0]/sum_rows[0]

        cm2 = cm1*100

        # plotting the predicted percentages of recall using seaborn

        plt.figure(figsize=(9, 7))
        sn.heatmap(cm2, annot=True, fmt='.2f',
                   xticklabels=original_labels, yticklabels=original_labels)
        plt.title('Predicted Percentages')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')

        plt.savefig('predicted_percentages.png')
        plt.close()

        # extracting the diagonal entries of the predicted percentages that means which label has been predicted with how much accuracy
        diagonal_entries_of_predicted_percentages = np.diagonal(cm2)

        # converting the diagonal entries of the predicted percentages to a dictionary
        predicted_percentages = {key: value for key, value in zip(
            original_labels, diagonal_entries_of_predicted_percentages)}
        print(predicted_percentages)

        # saving the perdicted percentage as a bar chart
        plt.figure(figsize=(15, 7))
        plt.bar(predicted_percentages.keys(), predicted_percentages.values())
        plt.xlabel('Keys')
        plt.ylabel('Values')
        plt.title('Bar Graph of Keys vs Values')

        # Display the graph
        plt.savefig('predicted_percentages_bar_chart.png')

        # output dictionary
        output_dict = {
            'cm': cm,
            'cm2': cm2,
            'predicted_percentages': predicted_percentages,
            'model evaluation': model_evaluation
        }

        # defining keys of confusion matrix and predicted percentages
        confusion_matrix_key = "prediction results/confusion_matrix.png"
        predicted_percentages_key = "prediction results/predicted_percentages.png"
        predicted_percentages_bar_chart_key = "prediction results/predicted_percentages_bar_chart.png"

        # saving the confusion matrix and predicted percentages to s3 bucket
        hook = S3Hook()
        with open('confusion_matrix.png', 'rb') as file:
            confusion_matrix_bytes = file.read()
        hook.load_bytes(
            bytes_data=confusion_matrix_bytes,
            key=confusion_matrix_key,
            bucket_name=ENV_BUCKET,
            replace=True
        )
        with open('predicted_percentages.png', 'rb') as file:
            confusion_matrix_bytes = file.read()
        hook.load_bytes(
            bytes_data=confusion_matrix_bytes,
            key=confusion_matrix_key,
            bucket_name=ENV_BUCKET,
            replace=True
        )

        with open('predicted_percentages_bar_chart.png', 'rb') as file:
            predicted_percentages_bytes = file.read()
        hook.load_bytes(
            bytes_data=predicted_percentages_bytes,
            key=predicted_percentages_key,
            bucket_name=ENV_BUCKET,
            replace=True
        )

        pickled_data_prediction_key = "prediction results/pickled_data_prediction.pkl"
        # Save the output dictionary as a pickle file
        pickled_data_prediction = pickle.dumps(output_dict)
        hook.load_bytes(
            bytes_data=pickled_data_prediction,
            key=predicted_percentages_bar_chart_key,
            bucket_name=ENV_BUCKET,
            replace=True
        )

        return {"file path of predicted output": pickled_data_prediction_key}

    # calling functions

    key_of_pre_processing_result = pre_processing_data()
    key_of_training_output = training(key_of_pre_processing_result)

    prediction_and_analyzing_output = prediction_and_analyzing(
        key_of_pre_processing_result, key_of_training_output)


sp_classifier_dag_v4_using_taskflow_api = sp_classifier_dag_v4_using_taskflow_api()
