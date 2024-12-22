import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_cv():
    """Extract text from CV file"""
    try:
        file_path = input("Enter the path to your CV file (txt or pdf): ").strip()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError("File does not exist")
            
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_path.endswith('.pdf'):
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                print("Installing PyPDF2...")
                os.system("pip install PyPDF2")
                from PyPDF2 import PdfReader
                
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + " "
            return text
        else:
            raise ValueError("Unsupported file format. Please use .txt or .pdf")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def simple_preprocess(text):
    """Simple text preprocessing without NLTK"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common English stop words
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                 "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
                 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
                 'then', 'once'}
    
    words = text.split()
    text = ' '.join(word for word in words if word not in stop_words)
    
    return text

def find_jobs(job_descriptions):
    """Find matching jobs based on CV"""
    if not job_descriptions:
        print("No job descriptions provided")
        return
        
    # Extract CV text
    cv_text = extract_text_from_cv()
    if not cv_text:
        return
        
    # Preprocess texts
    cv_text = simple_preprocess(cv_text)
    processed_jobs = [simple_preprocess(desc) for desc in job_descriptions]
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform([cv_text] + processed_jobs)
    
    # Calculate similarities
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Rank and display results
    ranked_jobs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    
    print("\nTop Job Matches Based on Your CV:")
    print("-" * 50)
    for idx, score in ranked_jobs[:5]:  # Show top 5 matches
        print(f"\nMatch Score: {score:.2%}")
        print(f"Job Description: {job_descriptions[idx]}")
        print("-" * 50)

if __name__ == "__main__":
    # Example job descriptions
    jobs = [
        "Senior Data Scientist: Python, ML algorithms, and big data technologies required.",
        "Full Stack Developer: JavaScript, React, Node.js, and database management.",
        "AI Engineer: Deep learning, TensorFlow, and computer vision expertise needed.",
        "Data Analyst: SQL, Excel, and data visualization skills required.",
        "Machine Learning Engineer: ML frameworks, model deployment, and API development.",
        "Software Engineer: Java, Python, and cloud computing skills needed."
    ]
    
    # Run the job matcher
    find_jobs(jobs)
