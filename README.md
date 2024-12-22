# Job_Finder

This Python script implements a job-matching application using machine learning concepts. It processes a user's CV file and compares its content with predefined job descriptions to recommend the best matches. The script uses TF-IDF vectorization to compute similarities between the CV and job descriptions and ranks the jobs based on their relevance.

Key Features:
File Input and Text Extraction:

The extract_text_from_cv function reads text from a .txt or .pdf file.
For .pdf files, the script uses the PyPDF2 library (installed automatically if missing).
Preprocessing:

The simple_preprocess function cleans and processes text by:
Converting to lowercase.
Removing special characters and stop words.
Trimming extra whitespace.
Job Matching:

The find_jobs function compares the processed CV text with a list of job descriptions:
Preprocesses the CV and job descriptions.
Uses TF-IDF (Term Frequency-Inverse Document Frequency) to transform text into numerical vectors.
Calculates cosine similarity to measure the relevance between the CV and job descriptions.
Recommendations:

The top 5 job matches are displayed with their similarity scores.
Example Usage:
The user provides the path to their CV file (either .txt or .pdf).
The script processes the file and compares it with job descriptions such as:
"Senior Data Scientist: Python, ML algorithms, and big data technologies required."
"Full Stack Developer: JavaScript, React, Node.js, and database management."
"AI Engineer: Deep learning, TensorFlow, and computer vision expertise needed."
...and others.
It ranks the jobs by relevance and prints the top matches.
Libraries Used:
os: For file handling.
re: For text cleaning and preprocessing.
scikit-learn: For TF-IDF vectorization and cosine similarity.
PyPDF2: For extracting text from PDF files (installed automatically if missing).
How to Run:
Save the script to a file (e.g., job_matcher.py).
Install required libraries using pip install scikit-learn PyPDF2 (if not already installed).
Run the script with python job_matcher.py.
Input the path to your CV file when prompted.
