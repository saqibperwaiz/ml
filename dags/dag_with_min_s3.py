from datetime import datetime, timedelta
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow import DAG

default_args = {
    'owner': 'Areeb',
    'retry': 5,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id = 'dag_with_min_s3_v01',
    start_date = datetime(2024,2,15),
    schedule_interval = '@daily',
    default_args = default_args
) as dag:
    task1 = S3KeySensor(
        task_id = 'sensor_minio_s3',
        bucket_name='airflow',
        bucket_key = 'dataset_after_training.csv',
        aws_conn_id = 'minio_s3'
    )
    