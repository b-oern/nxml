# nxml
Docker Base Image for Maschine Learning

Start Docker Image:
```
docker run --rm -it -p 7070:7070 ghcr.io/b-oern/nxml:main
```

Query Service with:
```
curl 'http://127.0.0.1/?type=nlp&text=Text%0to%20analyse'
```
```json
{
  "text": "Good morning",
  "ner": [{"start_position": 5, "end_position": 12, "text": "morning", "tag": "TIME"}],
  "success": true,
  "lang": "en",
  "words": ["Good", "morning"],
  "sentimnet": {"polarity": 0.7, "subjectivity": 0.6000000000000001},
  "sentimnet_polarity": 0.7,
  "sentimnet_subjectivity": 0.6000000000000001,
  "sentences": ["Good morning"]
}
```


```
docker run --rm -it nxml bash
```

## Build
```
docker build . -t nxml
```
