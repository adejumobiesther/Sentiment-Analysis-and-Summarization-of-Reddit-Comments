import praw
import pandas as pd
import re
from praw.models import MoreComments
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain import OpenAI, LLMChain, HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain import HuggingFacePipeline
import textwrap
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

user_agent = "Scrapper 1.0 by /u/SpecialistFar5246"
reddit = praw.Reddit(
        client_id="jSs1O4Fcysd8HNpJlwlBLg",
        client_secret="HVSi2GBcWO_GXpfTdJ313txdv8tr-w",
        user_agent=user_agent
    )

def get_top_posts(name_subreddit):
    posts = []
    ml_subreddit = reddit.subreddit(name_subreddit)
    for post in ml_subreddit.top(limit=10):
        posts.append(
            [post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
    posts = pd.DataFrame(posts, columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

    return posts

def get_hot_posts(name_subreddit):
    posts = []
    ml_subreddit = reddit.subreddit(name_subreddit)
    for post in ml_subreddit.hot(limit=10):
        posts.append(
            [post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
    posts = pd.DataFrame(posts, columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

    return posts

def get_controversial_posts(name_subreddit):
    posts = []
    ml_subreddit = reddit.subreddit(name_subreddit)
    for post in ml_subreddit.controversial(limit=10):
        posts.append(
            [post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
    posts = pd.DataFrame(posts, columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

    return posts

def get_comments(posts, n):
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    title = posts.title[n]
    comment_section = []
    submission = reddit.submission(posts.id[n])
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        text = re.sub(TEXT_CLEANING_RE, ' ', str(comment.body).lower()).strip()
        text = text.replace('\n', " ").replace("\\", "")
        print(text)
        comment_section.append(text)

    return title, comment_section

def preprocessed(comment_section):
    separator = "@@@@"
    max_word_count = 500
    word_count = 0
    concatenated_text = ""

    for comment in comment_section:
        words = comment.split()
        comment_word_count = len(words)

        if word_count + comment_word_count <= max_word_count:
            concatenated_text += comment + "."
            word_count += comment_word_count
        else:
            concatenated_text += separator + comment + "."
            word_count = comment_word_count

    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0, separator="@@@@")
    chunks = text_splitter.split_text(concatenated_text)

    BULLET_POINT_PROMPT = """Write a concise summary of the following:
    {text}:"""

    prompt_template = PromptTemplate(template=BULLET_POINT_PROMPT,
                                     input_variables=["text"])

    doc_store = [Document(page_content=text) for text in chunks]


    return chunks, prompt_template, doc_store

def davinci_model(chunks, prompt_template, doc_store, key):
    OPENAI_API_KEY = key
    HUGGINGFACEHUB_API_TOKEN = "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"

    llm_model = OpenAI(model_name="text-davinci-003", openai_api_key=OPENAI_API_KEY, temperature=0)

    # define the summarization chain
    summarization_chain = load_summarize_chain(llm=llm_model,
                                                chain_type="map_reduce")

    output_summary = summarization_chain.run(doc_store)

    wrapped_text = textwrap.fill(output_summary,
                                 width=100)

    return wrapped_text

def bart_model(chunks, prompt_template, doc_store):
    HUGGINGFACEHUB_API_TOKEN = "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"
    llm = HuggingFacePipeline.from_model_id(
        model_id="lidiya/bart-large-xsum-samsum",
        task="summarization",
        model_kwargs={"temperature": 0, "max_length": 512})

    # define the summarization chain
    summarization_chain = load_summarize_chain(llm=llm,
                                                chain_type="map_reduce")
    output_summary = summarization_chain.run(doc_store)

    wrapped_text = textwrap.fill(output_summary,
                                 width=100)
    return wrapped_text

def sentiment_analysis(comment_section):
    df = pd.DataFrame(comment_section, columns=['comments'])
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    task = 'sentiment'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)

    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    results = []
    for text in df['comments']:
        encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        results.append(scores.tolist())

    df['negative_score'] = [score[0] for score in results]
    df['neutral_score'] = [score[1] for score in results]
    df['positive_score'] = [score[2] for score in results]

    def get_sentiment_label(row):
        if row['positive_score'] > row['neutral_score'] and row['positive_score'] > row['negative_score']:
            return 'positive'
        elif row['negative_score'] > row['neutral_score'] and row['negative_score'] > row['positive_score']:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment'] = df.apply(get_sentiment_label, axis=1)

    return df





