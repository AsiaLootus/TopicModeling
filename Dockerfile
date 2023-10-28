FROM python:3.10

WORKDIR /usr/src/TopicExtractor
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]