FROM python:3.12

WORKDIR /app

COPY requirements.txt model_class1.py project_bk1.py project_inference.py ./
RUN pip install -r requirements.txt
RUN ["model_class.py", "project_inference.py"]

CMD [ "python", "./project_bk1.py" ]