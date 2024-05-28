# Resume-Analyzer

## Introduction
This Flask application analyzes the similarity between a job description and a resume. It allows users to upload a job description and a resume file (in PDF or DOCX format) and calculates the cosine similarity between the job description and the text of the resume.

## Components
### Flask Web Application
The application is built using the Flask framework, allowing for easy deployment and interaction via a web interface.

### Text Extraction
- **PDF Extraction:** Utilizes PyPDF2 to extract text from PDF files.
- **DOCX Extraction:** Utilizes the python-docx library to extract text from DOCX files.

### Text Preprocessing
- **Tokenization:** Uses NLTK's `word_tokenize` to tokenize the text.
- **Stopword Removal:** Removes common English stopwords using NLTK's stopwords corpus.
- **Lemmatization:** Applies word lemmatization to reduce words to their base or root form.

### Similarity Calculation
- **Cosine Similarity:** Calculates the cosine similarity between the preprocessed job description and resume text using TF-IDF vectorization.

### Data Storage
- **CSV File:** Stores the preprocessed resume text and similarity score in a CSV file for further analysis.

## Usage
1. Navigate to the home page of the application.
2. Enter the job description in the provided form.
3. Upload a resume file (in PDF or DOCX format).
4. Submit the form to calculate the similarity.
5. View the similarity score on the result page.

The preprocessed resume text and similarity score are stored in a CSV file named `resume_data.csv`.

## Dependencies
- Flask
- NLTK
- pandas
- python-docx
- PyPDF2

## How to Use
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask application using `python app.py`.
4. Access the application through a web browser at `http://localhost:5000`.
