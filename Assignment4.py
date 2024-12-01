import pymongo
import math
from collections import Counter, defaultdict
import re

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["search_engine_v2"]
terms_collection = db["terms"]
documents_collection = db["documents"]


documents = [
    "After the medication, headache and nausea were reported by the patient.",
    "The patient reported nausea and dizziness caused by the medication.",
    "Headache and dizziness are common effects of this medication.",
    "The medication caused a headache and nausea, but no dizziness was reported.",
]

def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text).lower()  
    words = text.split()
    unigrams = words
    bigrams = [" ".join(pair) for pair in zip(words, words[1:])]
    trigrams = [" ".join(triple) for triple in zip(words, words[1:], words[2:])]
    return unigrams + bigrams + trigrams


def build_inverted_index(documents):
    documents_collection.delete_many({})  
    terms_collection.delete_many({})  

    inverted_index = defaultdict(list)

    for doc_id, content in enumerate(documents, start=1):
        tokens = preprocess_text(content)
        token_counts = Counter(tokens)

        documents_collection.insert_one({"_id": doc_id, "content": content})

        for term, count in token_counts.items():
            inverted_index[term].append({"doc_id": doc_id, "tf": count})

    for term, postings in inverted_index.items():
        terms_collection.insert_one({"term": term, "postings": postings})

def compute_similarity(query):
    query_tokens = preprocess_text(query)
    query_vector = Counter(query_tokens)

    query_magnitude = math.sqrt(sum(val ** 2 for val in query_vector.values()))

    doc_scores = defaultdict(float)

    for term in query_tokens:
        term_entry = terms_collection.find_one({"term": term})
        if term_entry:  
            for posting in term_entry["postings"]:
                doc_id = posting["doc_id"]
                tf = posting["tf"]

                doc_scores[doc_id] += query_vector[term] * tf

    results = []
    for doc_id, score in doc_scores.items():
        doc_content = documents_collection.find_one({"_id": doc_id})["content"]

        doc_magnitude = math.sqrt(
            sum(
                (posting["tf"] ** 2)
                for term in query_tokens
                for posting in (
                    terms_collection.find_one({"term": term}) or {"postings": []}
                )["postings"]
                if posting["doc_id"] == doc_id
            )
        )

        if query_magnitude > 0 and doc_magnitude > 0:
            similarity = score / (query_magnitude * doc_magnitude)
        else:
            similarity = 0.0
        results.append((doc_content, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

def execute_queries(queries):
    for i, query in enumerate(queries, start=1):
        print(f"Query {i}: {query}")
        results = compute_similarity(query)
        for content, score in results:
            print(f"{content}, {score:.2f}")
        print()

build_inverted_index(documents)

queries = [
    "nausea and dizziness",
    "effects",
    "nausea was reported",
    "dizziness",
    "the medication",
]

execute_queries(queries)
