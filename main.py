# import os
# import logging
# import pandas as pd
# from flask import Flask, request, jsonify
# from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from groq import Groq
# import mysql.connector

# app = Flask(__name__)

# # Set up logging
# logging.basicConfig(level=logging.INFO)  # You can change this to DEBUG for more detailed logs
# logger = logging.getLogger(__name__)

# # Load your Groq API key
# GROQ_API_KEY = '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'
# client = Groq(api_key=GROQ_API_KEY)

# # Initialize Sentence Transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# csv_file_path = "kpis.csv"  # Path to your CSV file
# csv_loader = CSVLoader(file_path=csv_file_path)
# documents = csv_loader.load()
# logger.info(f"Number of documents loaded from CSV: {len(documents)}")

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
# split_documents = text_splitter.split_documents(documents)

# # Generate embeddings for the CSV data
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# # Check if the embeddings folder exists and is not empty
# if os.path.exists("embeddings") and os.listdir("embeddings"):
#     # If the folder exists and is not empty, load the vectorstore from the embeddings
#     vectorstore = Chroma(
#         persist_directory="embeddings",
#         embedding_function=embedding_function
#     )
# else:
#     # If the folder is empty or does not exist, create a new vectorstore
#     vectorstore = Chroma.from_documents(
#         documents=split_documents, 
#         embedding=embedding_function, 
#         persist_directory="embeddings"
#     )
#     vectorstore.persist()
# logger.info(f"Number of documents in vectorstore: {vectorstore._collection.count()}")

# # SQL Database connection details
# db_config = {
#     'user': 'root',
#     'password': 'S#uhas@123',
#     'host': 'localhost',  
#     'port': 3306, 
#     'database': 'kpis_db',
# }

# # Function to execute the SQL query and retrieve the row from the SQL database
# def execute_sql_query(query):
#     query = query.strip()
#     connection = mysql.connector.connect(**db_config)
#     try:
#         cursor = connection.cursor(dictionary=True)
        
#         # Execute the query
#         cursor.execute(query)
        
#         # Fetch the result
#         result = cursor.fetchall()
#         return result

#     except mysql.connector.Error as err:
#         logger.error(f"SQL Error: {err}")
#         return None

#     finally:
#         if connection.is_connected():
#             cursor.close()
#             connection.close()

# async def rag(query: str, contexts: list) -> str:
#     logger.info("RAG process started")
#     context_str = "\n".join(contexts)
#     logger.debug(context_str)
    
#     # Create the prompt for the language model
#     prompt = f"""
#     You have been provided with a SQL table named kpis which has the following structure:

#     CREATE TABLE kpis (
#     KPI VARCHAR(100),
#     DESCRIPTION TEXT,
#     TABLE_NAME VARCHAR(100),
#     COMBINED_DESCRIPTION TEXT
#     );

#     Your task is to generate a precise SQL query based on a given user query and a set of combined_description fields retrieved from a vector store. The SQL query should retrieve the relevant row(s) from the kpis table.

#     Instructions:
#     1) Analyze the User Query and Contexts: You will be given a user query and a list of combined_description fields from the kpis table. Your goal is to identify the most relevant combined_description that closely matches the user query.
#     2) Select the Correct Row: Based on the best-matched combined_description, generate a SQL query to retrieve the corresponding row(s) from the kpis table. Ensure that the COMBINED_DESCRIPTION field in the SQL query matches the best-matched description.
#     3) Choose the Appropriate Table: The TABLE_NAME within the combined_description might indicate different tables (e.g., Instant, Summarized, Weekly). Use this information to ensure that the SQL query is specific to the correct context.
#     4) Output Only the SQL Query: The final output should be a valid SQL query that can be executed directly on the kpis table. The output should not include any additional explanations or text.

#     Example:
#     If a combined_description context is:

#     TABLE: Summarized
#     COMBINED_DESCRIPTION: Total outgoing local calls made to external networks over the past 30 days, measured in minutes.

#     and this is identified as the best match for the user query, the output should be:

#     SELECT * FROM kpis WHERE COMBINED_DESCRIPTION = 'Total outgoing local calls made to external networks over the past 30 days, measured in minutes.';

#     Generate the SQL query based on the provided context.

#     Contexts:
#     {context_str}

#     Query:
#     {query}

#     Output:
#     SELECT * FROM kpis WHERE COMBINED_DESCRIPTION = '...';
#     """

#     # Generate answer using Groq
#     try:
#         llm = client.chat.completions.create(
#             messages=[
#                 {"role": "system", "content": "You are a highly skilled data retrieval system..."},
#                 {"role": "user", "content": prompt}
#             ],
#             model="llama3-70b-8192",
#             temperature=1,
#             max_tokens=200
#         )
#         response = llm.choices[0].message.content.strip()
#         logger.info("RAG process completed successfully")
#         return response
#     except Exception as e:
#         logger.error(f"Error during RAG process: {e}")
#         return ""

# async def retrieve(query: str) -> list:
#     logger.info(f"Retrieving documents for query: {query}")
    
#     try:
#         # Create query embedding
#         logger.debug(f"Number of documents in vectorstore: {vectorstore._collection.count()}")
        
#         # Perform similarity search in the vector store
#         results = vectorstore.similarity_search(query, k=10)
#         logger.info(f"Found {len(results)} matching documents")
        
#         # Extract the combined descriptions from the results
#         contexts = [result.page_content for result in results]
#         return contexts
#     except Exception as e:
#         logger.error(f"Error during document retrieval: {e}")
#         return []

# @app.route('/smsbot', methods=['POST'])
# async def send_sms():
#     received_message = request.json.get('payload', '')
#     logger.info(f"Received message: {received_message}")

#     try:
#         # Execute the RAG process
#         contexts = await retrieve(received_message)
#         sql_query = await rag(received_message, contexts)
        
#         # Execute the SQL query and fetch the result
#         result = execute_sql_query(sql_query)
        
#         return jsonify({"message": result})
#     except Exception as e:
#         logger.error(f"Error in SMS bot process: {e}")
#         return jsonify({"error": "An error occurred"}), 500

# if __name__ == "__main__":
#     logger.info("Starting the Flask application")
#     app.run(host='0.0.0.0', port=8000)

# from flask import Flask, request, jsonify
# import logging
# import requests
# from requests.exceptions import RequestException
# from werkzeug.exceptions import BadGateway

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(filename='app.log', level=logging.INFO,
#                     format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')

# @app.route('/hello', methods=['GET'])
# def hello_world():
#     app.logger.info('Successful GET request to /hello')
#     return "Hello, World!"

# @app.route('/echo', methods=['POST'])
# def echo():
#     try:
#         data = request.json
#         app.logger.info(f'Successful POST request to /echo with data: {data}')
#         return jsonify({"received": data})
#     except Exception as e:
#         error_msg = f"Error in /echo: {str(e)}"
#         app.logger.error(error_msg, exc_info=True)
#         return jsonify({"error": "Internal Server Error", "details": error_msg}), 500

# @app.route('/test-request/<path:url>', methods=['GET'])
# def test_request(url):
#     try:
#         response = requests.get(url, timeout=5)
#         response.raise_for_status()
#         return jsonify({"status": "success", "status_code": response.status_code})
#     except requests.Timeout:
#         error_msg = f"Timeout error when requesting {url}. The upstream server took too long to respond (> 5 seconds)."
#         app.logger.error(error_msg, exc_info=True)
#         raise BadGateway(description=error_msg)
#     except requests.ConnectionError:
#         error_msg = f"Connection error when requesting {url}. Could not establish a connection to the upstream server. The server might be down or unreachable."
#         app.logger.error(error_msg, exc_info=True)
#         raise BadGateway(description=error_msg)
#     except requests.HTTPError as e:
#         error_msg = f"HTTP error when requesting {url}: Received status code {e.response.status_code} from upstream server."
#         app.logger.error(error_msg, exc_info=True)
#         return jsonify({"error": error_msg, "upstream_status_code": e.response.status_code}), e.response.status_code
#     except RequestException as e:
#         error_msg = f"Request error when requesting {url}: {str(e)}. This could be due to network issues or problems with the upstream server."
#         app.logger.error(error_msg, exc_info=True)
#         raise BadGateway(description=error_msg)

# @app.errorhandler(BadGateway)
# def handle_bad_gateway(e):
#     response = jsonify({
#         "error": "Bad Gateway",
#         "message": str(e.description),
#         "code": 502,
#         "details": "This error typically occurs when the server, while acting as a gateway or proxy, received an invalid response from the upstream server it accessed in attempting to fulfill the request."
#     })
#     response.status_code = 502
#     app.logger.error(f'502 Bad Gateway: {e.description}')
#     return response

# @app.errorhandler(Exception)
# def handle_exception(e):
#     error_msg = f'Unhandled exception: {str(e)}'
#     app.logger.error(error_msg, exc_info=True)
#     return jsonify({
#         "error": "Internal Server Error",
#         "message": error_msg,
#         "details": "An unexpected error occurred on the server. Please check the server logs for more information."
#     }), 500

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=8000)

import os
import mysql.connector
from flask import Flask, request, jsonify

app = Flask(__name__)

# Fetch MySQL connection details from environment variables
db_config = {
    'user': 'root',
    'password': 'BRgHrHbOHKFqvwnbAhWhSzndgexyyHnk',
    'host': 'mysql.railway.internal',
    'port': 3306,  # Default MySQL port is 3306
    'database': 'railway',
}

# Function to execute the SQL query and retrieve the row from the SQL database
def execute_sql_query(query):
    connection = mysql.connector.connect(**db_config)
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchall()
        return result

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# A simple route to test the connection
@app.route('/testdb', methods=['GET'])
def test_db():
    query = "SELECT * FROM kpi WHERE kpi = 'Instant_og_call_count'"  # Simple query to test the connection
    result = execute_sql_query(query)
    if result:
        return jsonify({"status": "Database connected!", "result": result})
    else:
        return jsonify({"status": "Failed to connect to the database."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
