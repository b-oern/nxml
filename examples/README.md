

# Embeddings

Embedding werden für die Aufgaben aus einer JSON-Datei berechnet, der Aufbau ist wiefolgt:
```
{
    "jobs": [
        {
            "id": "example_job_1",
            "text": "This is a test."
        },
        ...
    ]
}
```

Folgendes Beispiel muss im Hauptverzeichnis des Repositories ausgeführt werden.
```
docker run --rm -v "$(pwd)"/output:/output -v "$(pwd)"/examples:/examples nxml sh create_embeddings /examples/embedding_job.json /output/result.json
```