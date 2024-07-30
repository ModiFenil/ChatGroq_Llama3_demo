import os
from flask import Flask, request, render_template, redirect, flash, jsonify
from werkzeug.utils import secure_filename
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from flask_socketio import SocketIO, emit
import time
import threading

# Load environment variables
load_dotenv()

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = './us_census/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024  # 3 MB limit

socketio = SocketIO(app)

# Load Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding():
    embeddings = OllamaEmbeddings()
    loader = PyPDFDirectoryLoader(UPLOAD_FOLDER)
    documents = loader.load()
    socketio.emit('progress', {'message': 'Documents loaded', 'progress': 20})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)
    socketio.emit('progress', {'message': 'Documents split', 'progress': 50})

    vectors = FAISS.from_documents(final_documents, embeddings)
    socketio.emit('progress', {'message': 'Vectors created', 'progress': 80})
    
    return vectors

def search_documents(prompt1):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = app.config['vectors'].as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': prompt1})
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and file.filename.endswith('.pdf'):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                flash('File successfully uploaded')
                threading.Thread(target=process_file).start()
                return redirect(request.url)
            else:
                flash('Invalid file type. Please upload a PDF file.')
                return redirect(request.url)
        
        prompt1 = request.form.get('prompt')
        if prompt1:
            if 'vectors' not in app.config:
                flash('Vector store DB is not ready yet. Please wait.')
                return redirect(request.url)
            response = search_documents(prompt1)
            response_time = time.process_time()
            return render_template('index.html', response=response['answer'], context=response["context"], response_time=response_time)
    
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        emit_progress('File uploaded. Processing...', 10)
        
        thread = threading.Thread(target=process_file)
        thread.start()
        
        return jsonify({'message': 'File upload complete. Processing...'})
    else:
        return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

def process_file():
    vectors = vector_embedding()
    app.config['vectors'] = vectors
    emit_progress('Vector store DB is ready', 100, done=True)

def emit_progress(message, progress, done=False):
    socketio.emit('progress', {'message': message, 'progress': progress, 'done': done})

if __name__ == '__main__':
    socketio.run(app, debug=True)
