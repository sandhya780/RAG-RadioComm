import requests
import faiss
import os
import re
import openai
import wikipediaapi
import nltk
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
# from langchain_community.embeddings import OpenAIEmbeddings
from itertools import chain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# nltk.download('stopwords')
# from nltk.corpus import stopwords

def fetch_wikipedia_article(title):
    url = f"https://en.wikipedia.org/wiki/{title}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Error fetching Wikipedia page: {response.status_code}")
        return None
    

def extract_headings_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    return [heading.text.strip().replace("[edit]","") for heading in headings if heading.text.strip()]

def create_embeddings(text_list, model="text-embedding-ada-002"):
    embeddings = []
    for text in text_list:
        text = text.replace("\n", " ")  # Remove newlines
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    return embeddings

def store_embeddings_in_faiss(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype(np.float32))
    return index

def chunk_article_by_headings(html_content, title):
    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = []
    current_chunk = []
    current_heading = None

    for element in soup.find_all(['h2', 'p', 'h3', 'h4', 'h5', 'h6']):
        if element.name.startswith('h'):
            if current_chunk:
                chunks.append((title, current_heading, ' '.join(current_chunk)))
                current_chunk = []
            current_heading = element.get_text(strip=True)
        else:
            current_chunk.append(element.get_text(strip=True))
    
    if current_chunk:
        chunks.append((title,  current_heading, ' '.join(current_chunk)))
    
    return chunks


def load_faiss_index(file_path):
    return faiss.read_index(file_path)

def query_faiss_index(index, query_embedding, k=5):
    distances, indices = index.search(np.array([query_embedding]).astype(np.float32), k)
    return distances, indices

def bert_ranking(query, documents, model):
    # Encode the query and documents
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(documents, convert_to_tensor=True)
    
    # Compute cosine similarity scores
    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)
    
    # Convert similarities to numpy array for easy handling
    similarities = similarities.cpu().numpy().flatten()
    return similarities


def keyword_matching(query, documents):
    vectorizer = CountVectorizer().fit([query] + documents)
    query_vector = vectorizer.transform([query]).toarray()
    doc_vectors = vectorizer.transform(documents).toarray()
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, doc_vectors)
    return similarities.flatten()


def heading_similarity(query, headings):
    vectorizer = TfidfVectorizer().fit_transform([query] + headings)
    vectors = vectorizer.toarray()
    query_vector = vectors[0].reshape(1, -1)
    heading_vectors = vectors[1:]
    similarities = cosine_similarity(query_vector, heading_vectors)
    return similarities.flatten()


def main_query_flow(user_query, faiss_index_path, article_chunks, model):
    # Create an embedding for the user query
    query_embedding = create_embeddings([user_query])[0]
    
    # Load the FAISS index
    index = load_faiss_index(faiss_index_path)
    
    # Query the FAISS index
    distances, indices = query_faiss_index(index, query_embedding)
    
    # Retrieve the top-k chunks
    top_chunks = [article_chunks[idx] for idx in indices[0]]
    top_chunks_texts = [chunk[2] for chunk in top_chunks]
    top_chunks_headings = [chunk[1] for chunk in top_chunks]
    top_chunks_titles = [chunk[0] for chunk in top_chunks]
    
    # Calculate BERT similarity scores
    bert_scores = bert_ranking(user_query, top_chunks_texts, model)

    # Calculate heading similarity scores
    heading_sim_scores = heading_similarity(user_query, top_chunks_headings)

    # Calculate keyword matching scores
    keyword_scores = keyword_matching(user_query, top_chunks_texts)
    
    
    # Combine scores with weights
    combined_scores = []
    faiss_weight = 0.5
    bert_weight = 0.5
    heading_weight = 0.7
    keyword_weight = 0.3

    
    for i in range(len(top_chunks)):
        combined_score = (faiss_weight * (1 - distances[0][i])) + (bert_weight * bert_scores[i]) + (keyword_weight * keyword_scores[i]) + (heading_weight * heading_sim_scores[i])
        combined_scores.append((combined_score, distances[0][i], bert_scores[i], heading_sim_scores[i], keyword_scores[i], top_chunks[i]))

    # Sort by combined score descending
    combined_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Display the top ranked chunks
    result_list, doc_list, heading_list, content_list, combined_score_list, other_score_list = [], [], [], [], [], []
    print(f"Top {len(combined_scores)} results for query '{user_query}':")
    for i, (combined_score, dist, bert_score, heading_sim, keyword_score, (title, heading, chunk)) in enumerate(combined_scores):
        result_list.append(i+1)
        doc_list.append(title)
        heading_list.append(heading.replace('[edit]', ''))
        modified_chunk = re.sub(r'\[.*?\]', '', chunk[:1000])
        content_list.append(modified_chunk)
        combined_score_list.append(round(combined_score,3))
        other_scores = f"BERT Score: {round(bert_score, 3)}, Heading Similarity: {round(heading_sim, 3)}, Keyword Score: {round(keyword_score, 3)}"
        other_score_list.append(other_scores)

        # print(f"Result {i+1}: (Combined Score: {combined_score}, Distance: {dist}, BERT Score: {bert_score}, Heading Similarity: {heading_sim}, Keyword Score: {keyword_score})")
        # print(f"Document: {title}")
        # print(f"Heading: {heading}")
        # print(f"Content: {chunk[:1000]}")  # Print first 1000 characters of the chunk
        # print("-" * 20)

    df = pd.DataFrame(result_list, columns=['Results'])
    df['Document'] = doc_list
    df['Heading'] = heading_list
    df['Content'] = content_list
    df['Score'] = combined_score_list
    df['Other scores'] = other_score_list
    print(df)
    return df



# user_query = "list the types of two way radio and types of wireless networks"


def rag_response_and_rank(user_query, faiss_index_path, article_chunks, bert_model):
    # article_titles = ["Radio", "Radio_wave", "Two-way_radio", "Wireless_network",
    #               "Amplitude_modulation", "Frequency_modulation", "International_Telecommunication_Union"]  
    # headings_list = []
    # article_embeddings_list = []
    # article_chunks = []
    # faiss_index = None

    # for title in article_titles:
    #     print(f"Processing article: {title}")
    #     article_content = fetch_wikipedia_article(title)
        
    #     if article_content:
    #         headings = extract_headings_from_html(article_content)
    #         headings_list.append(headings)
            
    #         chunks = chunk_article_by_headings(article_content, title)
    #         article_chunks.extend(chunks)
    #         chunk_texts = [chunk[1] for chunk in chunks]
    #         article_embeddings = create_embeddings(chunk_texts)
    #         article_embeddings_list.extend(article_embeddings)
            
    #         if faiss_index is None:
    #             faiss_index = store_embeddings_in_faiss(article_embeddings)
    #         else:
    #             faiss_index.add(np.array(article_embeddings).astype(np.float32))

    #         print(f"Headings for '{title}':")
    #         for heading in headings:
    #             print(f"- {heading}")
    #         print("-" * 40)

    #         print(f"Chunks for '{title}':")
    #         for i, (title, heading, chunk) in enumerate(chunks):
    #             print(f"Chunk {i+1} - {heading} from document {title}:")
    #             print(chunk[:200])  # Print first 200 characters of each chunk
    #             print("-" * 20)

    # faiss_index_path = 'article_faiss_index.index'
    # faiss.write_index(faiss_index, faiss_index_path)
    # bert_model = SentenceTransformer('all-MiniLM-L6-v2')

    df = main_query_flow(user_query, faiss_index_path, article_chunks, bert_model)
    print(df)
    return df


# if __name__ == "__main__":
#     rag_response_and_rank(user_query)
