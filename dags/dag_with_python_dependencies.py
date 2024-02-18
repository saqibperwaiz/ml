from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


default_args = {
    'owner': 'Areeb',
    'retry': 5,
    'retry_delay': timedelta(minutes=2)
}


def get_sklearn():
    import sklearn
    print(f"sklearn with version: {sklearn.__version__} ")


def get_matplotlib():
    import matplotlib
    print(f"matplotlib with version: {matplotlib.__version__}")


def get_tensorflow():
    import tensorflow as tf
    print(f"tensorflow with version: {tf.__version__}")


with DAG(
    default_args=default_args,
    dag_id="dag_with_python_dependencies_v06",
    start_date=datetime(2024, 2, 16),
    schedule_interval='@daily'
) as dag:
    task1 = PythonOperator(
        task_id='get_sklearn',
        python_callable=get_sklearn
    )

    task2 = PythonOperator(
        task_id='get_matplotlib',
        python_callable=get_matplotlib
    )
    task3 = BashOperator(
        task_id='just_testing_task3',
        bash_command='echo "hello, I am task 3 and will be executed after task1"'
    )
    task4 = PythonOperator(
        task_id='get_tensorflow',
        python_callable=get_tensorflow
    )

    task1 >> task3 >> task2 >> task4
