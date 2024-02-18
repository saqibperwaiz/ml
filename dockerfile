FROM apache/airflow:latest
COPY requirements.txt /requirements.txt
RUN pip install --user --upgrade pip
RUN pip install --user -r /requirements.txt
