import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer, util
from collections import Counter
import re
import config
config.api_key=API_Key
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

# sidebar
with st.sidebar:
    st.header("What is a Semantic Search Engine?")
    st.markdown("**[Semantic Search](https://medium.com/nlplanet/semantic-search-with-few-lines-of-code-490df1d53fd6)** allows retrieving documents from a corpus using a search query in a semantic way. This means that the search engine looks not only for exact text matches, but also for **overlapping semantic meaning** (e.g. synonyms and periphrases).")
    st.markdown("This is different from a **text-matching search engine**, which looks for exact text matches only.")
    st.header("How does semantic search work?")
    st.markdown("The idea behind semantic search is to [embed](https://machinelearningmastery.com/what-are-word-embeddings/) all the entries in your corpus, which can be sentences, paragraphs, or documents, into a **vector space**. At search time, the query is embedded into the same vector space and the **[closest vectors](https://en.wikipedia.org/wiki/Cosine_similarity)** from your corpus are found.")
    st.header("Useful libraries")
    st.markdown("""
    - [`sentence-transformers`](https://sbert.net/): Allows to easily use great pre-trained models for semantic search and has a fast implementation for finding nearest neighbors by cosine similarity.
    - [`faiss`](https://github.com/facebookresearch/faiss): Allows efficient similarity search and clustering of dense vectors.
    """)
    st.header("Useful links")
    st.markdown("""
    - [Semantic Search with Sentence Transformers](https://medium.com/nlplanet/semantic-search-with-few-lines-of-code-490df1d53fd6)
    - [Sentence Transformers cheatsheet](https://medium.com/nlplanet/two-minutes-nlp-sentence-transformers-cheat-sheet-2e9865083e7a)
    """)
    st.header("Who made this?")
    st.markdown("This project was made by Siddhartha Mahajan [Github](https://github.com/Siddhartha-Mahajan) [LinkedIn](https://www.linkedin.com/in/siddharthamahajan03/) using a template made by Fabio Chiusano [LinkedIn](https://www.linkedin.com/in/fabio-chiusano-b6a3b311b/)")

# main content
st.header("Semantic Search Engine")
st.markdown("This is a small demo project of a semantic search engine. Enter the text to be search from or upload a text file/csv. Enter the query in the query box.")




def split_into_sentences(text: str):
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace(",","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

st_text = st.text_input("Write your text here", max_chars=10000)
original_title = '<center> <p style=" font-size: 30px;">OR</p></center>'
st.markdown(original_title, unsafe_allow_html=True)
from io import StringIO

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    

    # To read file as string:
    st_text= stringio.read()


    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
  
  
 

  



n_top_tags = 20




# collapse option to see a comparison between different search engine types
with st.expander("Semantic search engine vs Text match search engine"):
    st.markdown("""
    Here's a brief comparison between them:
    - Generally, a semantic search engine works better than a text-matching search engine, as the latter (1) looks for only exact text matches between the articles and the query after some [text normalization](https://towardsdatascience.com/text-normalization-for-natural-language-processing-nlp-70a314bfa646) and (2) it doesn't take into account synonyms, etc.
    - The quality difference is higher if the corpus of articles is small (e.g. hundreds or thousands), because a text-matching search engine may return zero-or-few results for some queries, while a semantic search engine always returns an ordered list of articles.
    - On the other hand, a semantic search engine needs all the documents in the corpus to be embedded (i.e. transformed into semantic vectors thanks to a machine learning model) as a setup step, but this has to be done only once so it's not really a problem.
    - Using appropriate data structures that implement [fast approximate nearest neighbors algorithms](https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6), both types of search engines can have low latencies.
    """)


st_query = st.text_input("Write your query here", max_chars=10000)

def on_click_search():
    
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    sentences1=split_into_sentences(st_text) 
    sentences2=[st_query]
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    c=[]


    for i in range(len(sentences1)): c.append(cosine_scores[i][0])
    df=pd.DataFrame({'Sentences':sentences1,'Score':c})

    df1=df.sort_values('Score',ascending=False) 
    df1.set_index("Sentences",inplace=True)
    return df1  
session_state=True
st.button("Search", on_click=on_click_search)

if st_query != "":
    session_state = False
    on_click_search()


from serpapi import GoogleSearch
params = {
  "q": st_query,
  "location": "Delhi, India",
  "hl": "en",
  "gl": "in",
  "google_domain": "google.com",
  "api_key":API_Key
}
def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = link.split('=')[1]
    return f'<a target="_blank" href="{link}">{text}</a>'
def google_search():
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    df5=pd.DataFrame.from_dict(organic_results)
   
    df10=df5[['title','snippet','link']]
    return df10

if not session_state:
    st.markdown("### Results from uploaded text/document")#Output the pairs with their score
    df1=on_click_search()

    st.dataframe(df1.head(20),use_container_width=True)
    st.markdown("### Results from Google")
    df10=google_search()
    
    st.write(df10.to_html(escape=False, index=False), unsafe_allow_html=True)
    