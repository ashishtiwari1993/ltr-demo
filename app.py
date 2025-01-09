import streamlit as st
from elasticsearch import Elasticsearch
import yaml, os
import random, string
import hashlib
import pandas as pd

with open("./config/config.yml", "r") as file:
    config = yaml.safe_load(file)

es = Elasticsearch(
    cloud_id=config["elastic"]["cloud_id"], api_key=config["elastic"]["api_key"]
)

st.title("LTR demo")
st.header("Movies")


def getMovies(search_query="", ltr=""):

    q = {"query": {"match_all": {}}}

    if search_query:
        q = {"query": {"match": {"story": search_query}}}

    if ltr and search_query:
        ltr_rescorer = {
            "learning_to_rank": {
                "model_id": config["elastic"]["model_id"],
                "params": {"query": search_query},
            },
            "window_size": 100,
        }

        q["rescore"] = ltr_rescorer

    print(q)
    resp = es.search(index=config["elastic"]["index_name"], body=q)

    return resp["hits"]["hits"]


def get_short_hash(value, length=8):
    # Create a SHA256 hash of the string
    hash_object = hashlib.sha256(value.encode())
    # Get the hexadecimal digest
    full_hash = hash_object.hexdigest()
    # Return the first `length` characters of the hash
    return full_hash[:length]


judgement_list = "judgement_list.csv"

if not os.path.exists(judgement_list):
    jdf = pd.DataFrame(columns=["query_id", "query", "doc_id", "grade"])
    jdf.to_csv(judgement_list, index=False, header=True)


def push_to_csv(doc_id, relevancy, search_query=""):
    row_data = {
        "query_id": ["qid:" + get_short_hash(search_query)],
        "query": [search_query],
        "doc_id": [doc_id],
        "grade": [relevancy],
    }

    df = pd.DataFrame(row_data)
    df.to_csv(judgement_list, mode="a", index=False, header=False)


search_keyword = st.text_input(label="Search")

# resp = getMovies(search_keyword)
resp = getMovies(search_keyword, True)

for m in resp:

    st.header(m["_source"]["title_x"])
    st.image(m["_source"]["poster_path"])
    st.write(m["_source"]["story"])

    st.button(
        label="Relevant",
        key=m["_id"],
        args=[m["_id"], 1, search_keyword],
        on_click=push_to_csv,
    )
    st.button(
        label="Not Relevant",
        key=m["_id"] + random.choice(string.ascii_letters),
        args=[m["_id"] + random.choice(string.ascii_letters), 0, search_keyword],
        on_click=push_to_csv,
    )
