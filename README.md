# TopicModeling
Project of Short Text Topic Modeling

# Docker commands
docker build -t topic_extract .

docker run -d --name topic_container --env-file .env -v output:/usr/src/TopicExtractor/output topic_extract

# .env file example
OPENAI_KEY=sk-...

FILENAME=data/datasets/json/2.json

FILENAME_QUESTION=data/datasets/sq/2q.txt	
