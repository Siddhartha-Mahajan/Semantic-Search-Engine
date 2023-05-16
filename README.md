# Semantic Search Engine
Semantic Search allows retrieving documents from a corpus using a search query in a semantic way. This means that the search engine looks not only for exact text matches, but also for overlapping semantic meaning (e.g. synonyms and periphrases).
\
This repository contains code for my semantic search engine which can search for user provided queries on user provided text/documents and also on the internet.

## How did I go about it?
I did it using Streamlit. The text blob was broken down into sentences using regex [StackOverflow](https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences).\
The sentences were then encoded and converted to tensors using all-MiniLM-L6-v2 model under [SentenceTransformers](https://github.com/UKPLab/sentence-transformers). \
Then cosine-similarity scores are calculated with the user provided query (also tensorised) and these sentences.\
A Pandas dataframe is used to store these and the sentences with the top 20 scores are displayed.

## Google Search Results
Using SerpApi's Google Search API to get the top results' title, links and their snippets, I displayed these as a table. This was done mostly as a 'can be looked more into in the future' mindset and may be useful in some scenarios.

## Template
The Streamlit template I used was made by Fabio Chiusano [(link)](https://huggingface.co/spaces/fabiochiu/semantic-search-medium).  Huge thanks to him!

## Deployment
The app is deployed here: [Semantic-Search-Engine](https://huggingface.co/spaces/siddhartha-mahajan/Semantic-Search-Engine).

## To run it locally
1. Clone the repository
2. Set up a virtual env (Python)
3. run ``` $ pip install -r requirements.txt ```
5. run ``` $ streamlit run "Path-to-Repo\Semantic-Search-Engine\app.py" ```

## Video Demo
The video demo can be seen here: 
