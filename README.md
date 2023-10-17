# Tip of Your Tongue (ToYT)
ToYT is a RESTful API that helps you find the word you're looking for. 

## How it Works
To use ToYT, provide one or more sentences, each containing one blank token (`[MASK]`), and ToYT will return a list 
of words that could fill in the blank. Often a single sentence will be enough, but adding more sentences can improve
the results.

## Example
Given the following sentence:
```
The author has a/an ____ for creating vivid and memorable characters in his novels.
```

ToYT will return the following list of words:
```
affinity
knack
penchant
propensity
talent
```

## Usage
To use ToYT, send a POST request to the `/api/predict-word/` endpoint with the following JSON payload:
```json
{
  "sentences": [
    "The author has a/an ____ for creating vivid and memorable characters in his novels."
  ],
  "num_words": 5
}
```

ToYT will return a JSON response with the following structure:
```json
{
  "sentences": [
    {
      "sentence": "The author has a/an ____ for creating vivid and memorable characters in his novels.",
      "words": [
        "affinity",
        "knack",
        "penchant",
        "propensity",
        "talent"
      ]
    }
  ]
}
```