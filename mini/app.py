ENDPOINT = "us-central1-aiplatform.googleapis.com"
REGION = "us-central1"
PROJECT_ID = "saturam"
api_url = f"https://{ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import Config
import logging
from logging.handlers import RotatingFileHandler
from flask import request, jsonify
import psycopg2
from psycopg2 import sql
import services.utils as utils
from auth.middleware import api_key_required
import traceback
import requests
import json

# === New Imports for AI + Thread Safety ===
import os
import threading
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize extensions
limiter = Limiter(key_func=get_remote_address)

# === Thread-safe in-memory registry ===
endpoint_registry = {}
registry_lock = threading.Lock()

def get_access_token_from_service_account(service_account_file, scopes):
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=scopes
    )
        
    from google.auth.transport.requests import Request
    request = Request()
    credentials.refresh(request)
    
    return credentials.token



# === AI Analysis Function ===
def generate_ai_insights(query: str, data: list, analysis_type: str = "standard"):
    """Generate analysis using Llama API"""
    try:
        
        # Get access token
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        access_token = get_access_token_from_service_account(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'), scopes)
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        prompts = {
            "standard": f"""
            [INST] <<SYS>>
            You are a data analysis assistant. Analyze this database query and results:
            <</SYS>>
            
            Query: {query}
            Data Sample: {data[:3]} (showing first 3 of {len(data)} records)
            
            Provide:
            1. Key statistical insights
            2. Notable patterns/trends
            3. Data quality observations
            4. 3 actionable recommendations [/INST]
            """,
            "technical": f"""
            [INST] <<SYS>>
            You are a database optimization expert. Analyze this SQL query execution:
            <</SYS>>
            
            Query: {query}
            Returned {len(data)} records
            
            Provide:
            1. Query optimization suggestions
            2. Index recommendations
            3. Schema improvement ideas
            4. Potential performance bottlenecks [/INST]
            """,
            "business": f"""
            [INST] <<SYS>>
            You are a business intelligence analyst. Analyze these records:
            <</SYS>>
            
            Query: {query}
            Data Sample: {data[:5]}
            
            Provide:
            1. Key business insights
            2. Revenue/profit opportunities
            3. Risk factors
            4. Strategic recommendations [/INST]
            """
        }

        prompt = prompts.get(analysis_type, prompts["standard"])
        
        # Fixed payload structure - removed stream and fixed variable name
        payload = {
            "model": "meta/llama-3.1-405b-instruct-maas",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.3
        }

        
        response = requests.post(api_url, headers=headers, json=payload)
        
    
        response.raise_for_status()
        
        # Parse the OpenAI-compatible response format
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            # Try different possible response fields
            choice = result["choices"][0]
            content = (choice.get("message", {}).get("content") or 
                      choice.get("text") or 
                      "No response from AI model")
            return content
        else:
            return f"Unexpected response format: {result}"
        
    except requests.exceptions.RequestException as e:
        return f"Request to Llama API failed: {str(e)}"
    except Exception as e:
        return f"AI analysis failed: {str(e)}"


# === App Factory ===
def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Configure extensions
    limiter.init_app(app)
    
    # Configure logging
    handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)

    # === Preload cache ===
    utils.preload_registered_endpoints()

    app.config['LLAMA_API_CONFIG'] = {
        'endpoint': api_url,
        'max_tokens': 1024,
        'temperature': 0.3
    }

    @app.route('/generate-endpoint', methods=['POST'])
    @limiter.limit(app.config['RATE_LIMIT'])
    @api_key_required
    def generate_endpoint():
        """Generates a dynamic endpoint and stores the query in the database."""
        try:
            data = request.get_json()

            table_name = data.get('table_name')
            schema_name = data.get('schema_name', 'public')
            filter_columns = data.get('filter_columns', [])
            select_columns = data.get('select_columns', [])
            created_by = data.get('created_by', 'anonymous')

            if not table_name or not select_columns:
                return jsonify({'error': 'Both table_name and select_columns are required'}), 400

            schema = utils.get_table_schema(schema_name, table_name)
            if not schema:
                return jsonify({'error': f'Table {table_name} not found'}), 404

            for col in select_columns + filter_columns:
                if col not in schema:
                    return jsonify({'error': f'Column {col} not found in table {table_name}'}), 400

            endpoint_name = utils.generate_endpoint_name(table_name, select_columns, filter_columns)
            endpoint_path = f'/dynamic/{endpoint_name}'

            if utils.endpoint_exists(endpoint_name):
                return jsonify({'Info': f'Endpoint already exists at {endpoint_path}'}), 400

            query_str = utils.make_dynamic_query(schema_name, table_name, select_columns, filter_columns)
            created_at = utils.get_current_datetime()
            utils.insert_endpoint_to_db(endpoint_name, endpoint_path, query_str, created_by, created_at)

            # === Register endpoint in thread-safe dictionary ===
            with registry_lock:
                endpoint_registry[endpoint_name] = {
                    "query": query_str,
                    "path": endpoint_path,
                    "config": data
                }

            return jsonify({
                'message': f'Endpoint created at {endpoint_path}',
                'endpoint_path': endpoint_path,
                'method': 'GET',
                'ai_capabilities': {
                    "available": True,
                    "usage": "Add ?ai_analyze=true to request",
                    "analysis_types": ["standard", "technical", "business"]
                }
            })
        
        except Exception as e:
            app.logger.error(traceback.format_exc())
            app.logger.error('Dynamic endpoint Generation API Error: {}'.format(e))
            return jsonify({"error": "Internal server error"}), 500

    @app.route('/dynamic/<path:endpoint_name>', methods=['GET'])
    @limiter.limit(app.config['RATE_LIMIT'])
    @api_key_required
    def handle_dynamic_endpoint(endpoint_name):
        try:
            if not utils.endpoint_exists(endpoint_name):
                return jsonify({'error': f'Endpoint /{endpoint_name} not found'}), 404

            filters = request.args.to_dict()

            # Remove keys not intended for SQL WHERE clause
            non_sql_keys = ['ai_analyze', 'analysis_type']
            filtered_params = {k: v for k, v in filters.items() if k not in non_sql_keys}

            filter_values = list(filtered_params.values())
            result = utils.get_query_by_endpoint_name(endpoint_name)
            query_template = result["query"]
            endpoint_id = result["id"]

            if not query_template:
                return jsonify({'error': 'No query found for this endpoint'}), 404

            final_query = sql.SQL(query_template)
            conn = psycopg2.connect(**utils.DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(final_query, filter_values)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            data = [dict(zip(columns, row)) for row in rows]

            ai_analysis = None
            if request.args.get('ai_analyze', '').lower() in ('true', '1', 'yes'):
                analysis_type = request.args.get('analysis_type', 'standard')
                ai_analysis = generate_ai_insights(final_query.as_string(conn), data, analysis_type)

            response = {
                "metadata": {
                    "endpoint": endpoint_name,
                    "query": final_query.as_string(conn),
                    "record_count": len(data)
                },
                "data": data
            }

            if ai_analysis:
                response["ai_analysis"] = ai_analysis

            return jsonify(response)

        except Exception as e:
            app.logger.error(traceback.format_exc())
            app.logger.error('Dynamic endpoint Error: {}'.format(e))
            return jsonify({'error': 'Internal server error'}), 500
        finally:
            cursor.close()
            conn.close()

    # === Endpoint to list registered endpoints (optional) ===
    @app.route('/list-endpoints', methods=['GET'])
    @api_key_required
    def list_registered_endpoints():
        with registry_lock:
            return jsonify({
                "registered_endpoints": list(endpoint_registry.keys()),
                "count": len(endpoint_registry)
            })

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, threaded=True)