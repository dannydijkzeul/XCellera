import argparse
from json.tool import main
import os

# from xml.dom.minidom import Document

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"



from google import genai
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex

# from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.vertex_endpoint import VertexEndpointEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.vertex import Vertex




project_id = "dw-access-product"
location_id = "europe-west4"


def main(query: str):

    Settings.llm = Vertex(
        model="gemini-2.0-flash", project=project_id, location = location_id
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    print("model set")

    try:
        corpus = SimpleDirectoryReader("./corpus/").load_data()
        print("data loaded")
    
        index = VectorStoreIndex.from_documents(corpus)
        print("index done")

        query_engine = index.as_query_engine()

        response = query_engine.query(query)

        print("----Vertex AI response---")
        print(response)

    except Exception as e:
        print("broken")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="query"
    )


    parser.add_argument(
        "--query", type=str, required=True
    )

    args = parser.parse_args()

    main(args.query)

# client = genai.Client(
#     vertexai=True,
#     project='dw-access-product',   # Switch to your division
#     location='europe-west4'
    
# )
# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="Write a short story of 10 lines.",
# )
# print(response.text)