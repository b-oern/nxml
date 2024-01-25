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


```
docker run --rm -it nxml bash
```

## Build
```
docker build . -t nxml
```
