FROM python:3.8
FROM tensorflow/tensorflow

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . . 

CMD [ "python", "./main.py" ] .