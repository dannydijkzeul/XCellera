import os
from flask import Flask, request, render_template
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.vertex import Vertex

# Setup environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

# Vertex AI setup
project_id = "dw-access-product"
location_id = "europe-west4"

# Initialize app
app = Flask(__name__)

# Global variables (load once and reuse)
query_engine = None
# base_query_prompt = """
# Context: You are an agent that supports in responding to RFP questions.
# You work for Cover Genius, an Insurtech. The query below needs to be answered and it needs to get one of the 
# following classifications: insurance, tech, general, legal, other. The goal is to answer the questions and inform the prospect of the
# capabalities of Cover Genius and its products.
# Make the information specific to the specified partner. 
# If you find a lot of information please provide a long answer.
# Response format: <classifcation>;<rfp response> 
# Query:"""

base_query_prompt = """
Context: You are an agent that supports in responding to RFP questions.
You work for Cover Genius, an Insurtech. The query below needs to be answered and it needs to get one of the 
following classifications: insurance, tech, general, legal, other. The goal is to answer the questions and inform the prospect of the
capabalities of Cover Genius and its products.
If you find a lot of information please provide a long answer.
Response format: <classifcation>;<rfp response> 
Query:"""

# Load models and index ONCE when app starts
def initialize():
    global query_engine

    # Initialize LLM + embedding model
    Settings.llm = Vertex(
        model="gemini-2.0-flash", project=project_id, location=location_id
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Settings.chunk_size = 256
    # Settings.chunk_overlap = 25

    # Load and index corpus
    corpus = SimpleDirectoryReader("./corpus/").load_data()
    index = VectorStoreIndex.from_documents(corpus)
    print("Corpus loaded")

    # Create query engine (cached)
    query_engine = index.as_query_engine()

# Route
@app.route("/", methods=["GET", "POST"])
def index():
    print("flask started")
    response = None
    classification = None
    partner = ""
    query = base_query_prompt

    if request.method == "POST":
        user_input = request.form.get("query")
        partner = request.form.get("partner")
        # full_query = query + " " + user_input + " partner: " + partner
        full_query = query + " " + user_input
        try:
            result = query_engine.query(full_query)
            classification, response = result.response.split(";", 1)
        except Exception as e:
            classification = "Error"
            response = f"Something went wrong: {e}"

    return render_template("index.html", query=query, classification=classification, response=response)

# Start app with initialization
if __name__ == "__main__":
    initialize()
    app.run(debug=True)
