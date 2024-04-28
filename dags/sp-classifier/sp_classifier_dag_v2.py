# DAG exhibiting task flow paradigm in airflow 2.0
# https://airflow.apache.org/docs/apache-airflow/2.0.2/tutorial_taskflow_api.html
# Modified for our use case

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
from datetime import timedelta
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.S3_hook import S3Hook
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.hooks.subprocess import SubprocessHook
import sys
import subprocess
default_args = {
    'owner': 'airflow',
    'retry': 5,
    'retry_delay': timedelta(minutes=2)
}


@dag(default_args=default_args, schedule_interval="@daily", start_date=days_ago(2), tags=['SP_Classifier'])
def sp_classifier_dag_v2_using_taskflow_api():

    @task
    def installing_dependencies_using_subprocess():

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "numpy"])
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pandas"])
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "seaborn"])
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "scikit-learn"])
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "tensorflow"])
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "joblib"])
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "matplotlib"])
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "s3fs"])

        return {"message": "Dependencies installed successfully"}

    @task
    def print_request_version():
        import requests
        print("Requests version:", requests.__version__)

    @task
    def print_seaborn_version(message):
        print(message)
        import seaborn as sn
        print("Seaborn version:", sn.__version__)

    @task
    def read_file():
        hook = S3Hook()
        file_content = hook.read_key(
            key="customer.csv", bucket_name="my-landing-bucket"
        )
        return file_content

    @task
    def printCurrent_directory():
        import os
        print(os.listdir())

    @task()
    def pre_processing_data(dependecies_message):
        import os
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        import joblib
        from sklearn import preprocessing
        from sklearn.preprocessing import normalize
        import pickle
        from sklearn.preprocessing import MinMaxScaler

        df = pd.read_csv('data/dataset_after_training.csv')

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
        file_path = "data/preprocessing_output.pkl"

        pickled_data = pickle.dumps(my_dict)

        with open(file_path, 'wb') as file:
            file.write(pickled_data)

    # Return the file path
        return file_path

    @task(multiple_outputs=True)
    def training(dependecies_message, file_path_of_preprocessed_data):

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

        with open(file_path_of_preprocessed_data, 'rb') as file:
            pickled_data = file.read()

        # Unpickle the data
        unpickled_data = pickle.loads(pickled_data)
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

        file_path_of_trained_model = "data/trained_model.pkl"

        pickled_data = pickle.dumps(model)

        with open(file_path_of_trained_model, 'wb') as file:
            file.write(pickled_data)

        return {"file path of trained model": file_path_of_trained_model}

    @task()
    def prediction_and_analyzing(dependecies_message, file_path_of_preprocessed_data):
        import joblib
        import pickle
        import numpy as np
        import seaborn as sn
        import matplotlib.pyplot as plt
        import tensorflow as tf

        """
        #### Load task
        A simple Load task which takes in the result of the Transform task and
        instead of saving it to end user review, just prints it out.
        """

        with open(file_path_of_preprocessed_data, 'rb') as file:
            pickled_data = file.read()

        # Unpickle the data
        unpickled_data = pickle.loads(pickled_data)

        normalized_features = unpickled_data['normalized_features']
        y_test = unpickled_data['y_test']
        encoded_scaler_label = unpickled_data['encoded_scaler_label']
        categorical_labels = unpickled_data['categorical_labels']

        model = joblib.load("data/trained_model.pkl")

        predicted_normalized_features = model.predict(normalized_features)
        predicted_label_full_dataset = [
            np.argmax(i) for i in predicted_normalized_features]

        model.evaluate(normalized_features, categorical_labels)

        cm = tf.math.confusion_matrix(
            labels=categorical_labels, predictions=predicted_label_full_dataset)

        original_labels = encoded_scaler_label.inverse_transform(
            np.unique(categorical_labels))

        plt.figure(figsize=(9, 7))
        sn.heatmap(cm, annot=True, fmt='d',
                   xticklabels=original_labels, yticklabels=original_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')

        # saving the confusion matrix
        plt.savefig('data/confusion_matrix.png')
        plt.close()

        sum_rows = np.sum(cm, axis=1)
        cm1 = tf.convert_to_tensor(
            np.array([cm[i]/sum_rows[i] for i in range(len(cm))]))
        # cm1 = cm[0]/sum_rows[0]

        cm2 = cm1*100

        plt.figure(figsize=(9, 7))
        sn.heatmap(cm2, annot=True, fmt='.2f',
                   xticklabels=original_labels, yticklabels=original_labels)
        plt.title('Predicted Percentages')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')

        plt.savefig('data/predicted_percentages.png')
        plt.close()

        diagonal_entries_of_predicted_percentages = np.diagonal(cm2)

        predicted_percentages = {key: value for key, value in zip(
            original_labels, diagonal_entries_of_predicted_percentages)}
        print(predicted_percentages)

        # saving the perdicted percentage as a bar chart
        plt.figure(figsize=(15, 7))
        plt.bar(predicted_percentages.keys(), predicted_percentages.values())

        # Add labels and a title
        plt.xlabel('Keys')
        plt.ylabel('Values')
        plt.title('Bar Graph of Keys vs Values')

        # Display the graph
        plt.savefig('data/predicted_percentages_bar_chart.png')
        plt.close()

        output_dict = {
            'cm': cm,
            'cm2': cm2,
            'predicted_percentages': predicted_percentages
        }

        # Save the output dictionary as a pickle file
        pickled_data = pickle.dumps(output_dict)
        file_path_of_predicted_output = "data/model_predicted_output.pkl"
        with open(file_path_of_predicted_output, 'wb') as file:
            file.write(pickled_data)

        return {"file path of predicted output": file_path_of_predicted_output}

    message = installing_dependencies_using_subprocess()
    print_request_version()
    print_seaborn_version(message)
    pre_processing_result = pre_processing_data(message)
    model = training(message, pre_processing_result)

    prediction_and_analyzing_output = prediction_and_analyzing(message,
                                                               pre_processing_result)


sp_classifier_dag_v2_using_taskflow_api = sp_classifier_dag_v2_using_taskflow_api()
