from flask import Flask, render_template, request
import pandas as pd
import sys

from RAG_BERT import *  # Replace 'your_module' with the actual module name

app = Flask(__name__)


article_titles = ["Radio", "Radio_wave", "Two-way_radio", "Wireless_network",
                  "Amplitude_modulation", "Frequency_modulation", "International_Telecommunication_Union"]  
headings_list = []
article_embeddings_list = []
article_chunks = []
faiss_index = None

for title in article_titles:
    print(f"Processing article: {title}")
    article_content = fetch_wikipedia_article(title)
    
    if article_content:
        headings = extract_headings_from_html(article_content)
        headings_list.append(headings)
        
        chunks = chunk_article_by_headings(article_content, title)
        article_chunks.extend(chunks)
        chunk_texts = [chunk[1] for chunk in chunks]
        article_embeddings = create_embeddings(chunk_texts)
        article_embeddings_list.extend(article_embeddings)
        
        if faiss_index is None:
            faiss_index = store_embeddings_in_faiss(article_embeddings)
        else:
            faiss_index.add(np.array(article_embeddings).astype(np.float32))

        # print(f"Headings for '{title}':")
        # for heading in headings:
        #     print(f"- {heading}")
        # print("-" * 40)

        # print(f"Chunks for '{title}':")
        # for i, (title, heading, chunk) in enumerate(chunks):
        #     print(f"Chunk {i+1} - {heading} from document {title}:")
        #     print(chunk[:20])  # Print first 20 characters of each chunk
        #     print("-" * 20)

faiss_index_path = 'article_faiss_index.index'
faiss.write_index(faiss_index, faiss_index_path)
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/', methods=['GET', 'POST'])
def home():
    results = None
    if request.method == 'POST':
        try:
            user_query = request.form.get('query')
            print(f"Received query: {user_query}")  # Debugging line
            df = rag_response_and_rank(user_query, faiss_index_path, article_chunks, bert_model)  # Call the external function
            print(f"DataFrame:\n{df}")  # Debugging line
            results = df.to_dict(orient='records')  # Convert DataFrame to list of dictionaries for easier rendering
        except Exception as e:
            print(f"Error processing query: {e}", file=sys.stderr)
            results = [{'Error': str(e)}]  # Display error in the table
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=False)
