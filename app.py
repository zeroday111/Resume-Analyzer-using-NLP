from flask import Flask, render_template, request
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from docx import Document
from PyPDF2 import PdfFileReader
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
        return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

# Function to extract text from file
def extract_text_from_file(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format")

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Define stop words
    stop_words = set(stopwords.words('english'))
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a string
    clean_text = ' '.join(tokens)
    return clean_text

# Function to calculate cosine similarity
def calculate_similarity(job_description, resume_text):
    # Preprocess the job description and resume text
    job_description_cleaned = preprocess_text(job_description)
    resume_text_cleaned = preprocess_text(resume_text)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the data
    tfidf_matrix = vectorizer.fit_transform([job_description_cleaned, resume_text_cleaned])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    return cosine_sim[0][0]

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload and similarity calculation
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Get job description from form
        job_description = request.form['job_description']

        # Get uploaded file
        file = request.files['file']
        file_path = "uploaded_file." + file.filename.split(".")[-1]
        file.save(file_path)

        # Extract text from file
        resume_text = extract_text_from_file(file_path)

        # Preprocess the resume text
        cleaned_resume_text = preprocess_text(resume_text)

        # Calculate similarity
        similarity = calculate_similarity(job_description, resume_text)
        
        # Store resume text and similarity score in a DataFrame
        df = pd.DataFrame({'Resume Text': [cleaned_resume_text], 'Similarity Score': [similarity]})
        
        # Check if the CSV file already exists
        try:
            existing_data = pd.read_csv('resume_data.csv')
            updated_data = pd.concat([existing_data, df], ignore_index=True)
            updated_data.to_csv('resume_data.csv', index=False)
        except FileNotFoundError:
            df.to_csv('resume_data.csv', index=False)

        return render_template('result.html', similarity=similarity)

if __name__ == '__main__':
    app.run(debug=True)
