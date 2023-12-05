import streamlit as st
import re
import yake
from multi_rake import Rake
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from transformers.pipelines import pipeline
from flair.embeddings import TransformerDocumentEmbeddings, WordEmbeddings, DocumentPoolEmbeddings
from better_profanity import profanity
import spacy

#loading landguage models
kw_model3 = KeyBERT(model='all-mpnet-base-v2')
hf_model = pipeline(task = "feature-extraction", model="distilbert-base-cased")
kw_model4 = KeyBERT(model=hf_model)
try:
    spacy_model = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
except OSError:
    print('Downloading language model for the spaCy POS tagger\n'
        "(don't worry, this will only happen once)", file=stderr)
    from spacy.cli import download
    download('en')
    spacy_model = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
#spacy_model = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
kw_model5 = KeyBERT(model=spacy_model)
# header
st.title(':blue[Keyphrase Extractor - Qodequay]')

# Model input
model = st.selectbox(
    'Select a model for keyphrase extraction',
    ('KW_Gen1', 'KW_Gen2','KW_Gen3','KW_Gen4','KW_Gen5'))

def clear_text():
    st.session_state["text"] = ""

# Paragraph input
input_paragraph = st.text_area("Input Paragraph: ",key="text")
st.button("Clear", on_click=clear_text)
max_words = st.number_input("Max words in a keyphrase: ", min_value=1, max_value=4, step=1, value=2)
no_of_keywords = st.number_input("Number of keyprases required: ", min_value=3, max_value=12, step=1, value = 6)
input_paragraph = re.sub(r"[^a-zA-Z0-9 ]", "", input_paragraph)
input_paragraph = profanity.censor(input_paragraph,'')

if input_paragraph:
    if model == 'KW_Gen1':
        kw_extractor = yake.KeywordExtractor(lan="en", n=max_words, dedupLim=0.5, dedupFunc="seqm", windowsSize=1, top=no_of_keywords, features=None, stopwords=None)
        keywords = kw_extractor.extract_keywords(input_paragraph)
        kw_list = []
        for kw, v in keywords:
            kw_list.append(kw)
        # Display the keywords
        st.write("Keyphrases List: ", " ; ".join(kw_list))
    if model == 'KW_Gen2':
        rake = Rake(max_words=max_words)
        keywords = rake.apply(input_paragraph)
        kw_list = []
        for kw, v in keywords[:no_of_keywords]:
            kw_list.append(kw)
        # Display the keywords
        st.write("Keyphrases List: ", " ; ".join(kw_list))
    if model == 'KW_Gen3':
        keywords = kw_model3.extract_keywords(input_paragraph, top_n=no_of_keywords, keyphrase_ngram_range=(1, max_words), diversity=0.5, use_mmr=True, stop_words="english",highlight=False)
        kw_list = []
        for kw, v in keywords[:no_of_keywords]:
            kw_list.append(kw)
        # Display the keywords
        st.write("Keyphrases List: ", " ; ".join(kw_list))
    if model == 'KW_Gen4':
        keywords = kw_model4.extract_keywords(input_paragraph, top_n=no_of_keywords, keyphrase_ngram_range=(1, max_words), diversity=0.5, use_mmr=True, stop_words="english",highlight=False)
        kw_list = []
        for kw, v in keywords[:no_of_keywords]:
            kw_list.append(kw)
        # Display the keywords
        st.write("Keyphrases List: ", " ; ".join(kw_list))
    if model == 'KW_Gen5':
        keywords = kw_model5.extract_keywords(input_paragraph, top_n=no_of_keywords, keyphrase_ngram_range=(1, max_words), diversity=0.5, use_mmr=True, stop_words="english",highlight=False)
        kw_list = []
        for kw, v in keywords[:no_of_keywords]:
            kw_list.append(kw)
        # Display the keywords
        st.write("Keyphrases List: ", " ; ".join(kw_list))           
    if model == 'KW_Gen6':
        keywords = kw_model6.extract_keywords(input_paragraph, top_n=no_of_keywords, keyphrase_ngram_range=(1, max_words), diversity=0.5, use_mmr=True, stop_words="english",highlight=False)
        kw_list = []
        for kw, v in keywords[:no_of_keywords]:
            kw_list.append(kw)
        # Display the keywords
        st.write("Keyphrases List: ", " ; ".join(kw_list))

