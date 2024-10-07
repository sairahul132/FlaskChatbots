from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__)


# Function to scrape all content from the website
def scrape_website():
    url = 'https://www.javatpoint.com/python-tutorial'  # Replace with your website URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    content_dict = []

    # Scrape all text from relevant tags, including headers, paragraphs, lists, and other tags
    for tag in ['h1', 'h2', 'h3', 'p', 'li', 'ul', 'strong', 'span', 'div']:  # Added more tags
        for element in soup.find_all(tag):
            content = element.get_text().strip()
            if content:
                content_dict.append(content)

    return content_dict


# Load the SentenceTransformer model for embedding the questions and content
model = SentenceTransformer('all-MiniLM-L6-v2')

# Scrape website content and store it in memory
website_content = scrape_website()


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()

    if 'question' not in data:
        return jsonify({'error': 'No question provided!'}), 400

    user_question = data['question'].strip()

    # Encode both the user question and the website content into vectors
    question_embedding = model.encode(user_question, convert_to_tensor=True)
    content_embeddings = model.encode(website_content, convert_to_tensor=True)

    # Compute cosine similarities between the question and all content
    similarities = util.pytorch_cos_sim(question_embedding, content_embeddings)[0]

    # Get the indices of the most relevant answers, sorted by similarity
    top_n = 3  # You can adjust how many relevant sections you want to retrieve
    top_indices = similarities.argsort(descending=True)[:top_n]

    # Prepare a list of answers
    answers = []
    for index in top_indices:
        best_match_score = similarities[index].item()
        if best_match_score > 0.3:  # Adjust this threshold as needed
            matched_content = website_content[index]
            answers.append({matched_content})

    # If no good matches found, return an error message
    if not answers:
        answer = "I'm sorry, I couldn't find a suitable answer to your question."
    else:
        # Combine the top answers into a single response
        answer = "Here are the most relevant answers I found:\n" + "\n\n".join(answers)

    return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(debug=True)
