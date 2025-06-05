ENDPOINT = "us-central1-aiplatform.googleapis.com"
REGION = "us-central1"
PROJECT_ID = "saturam"
API_URL = f"https://{ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import Config
import logging
from typing import Dict, List, Optional
from logging.handlers import RotatingFileHandler
from flask import request, jsonify
import psycopg2
from psycopg2 import sql
import services.utils as utils
from auth.middleware import api_key_required
import traceback
import requests
import json
from google.auth.transport.requests import Request
from jsonschema import validate, ValidationError
from functools import wraps


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

def api_error_handler(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except psycopg2.Error as e:
            app.logger.error(f"Database error: {str(e)}")
            return jsonify({"error": "Database operation failed"}), 500
        except requests.exceptions.RequestException as e:
            app.logger.error(f"API request failed: {str(e)}")
            return jsonify({"error": "External API request failed"}), 502
        except ValidationError as e:
            app.logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": "Data validation failed"}), 400
        except Exception as e:
            app.logger.error(f"Unexpected error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    return wrapper

# JSON Schema for AI response validation
AI_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["key_findings", "potential_issues", "recommendations"],
    "properties": {
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "potential_issues": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "type_specific_insights": {"type": "array", "items": {"type": "string"}}
    }
}

def preload_registered_endpoints():
    try:
        conn = psycopg2.connect(**utils.DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT endpoint_name, query, endpoint_path FROM de_dynamic_api.endpoint_registry")  # Change table name if needed
        rows = cursor.fetchall()
        print(f"Preloading {len(rows)} endpoints from DB...")
        with registry_lock:
            for name, query, path in rows:
                endpoint_registry[name] = {
                    "query": query,
                    "path": path
                }
        print(endpoint_registry)
        cursor.close()
        conn.close()
    except Exception as e:
        print("Failed to preload endpoints from DB:", str(e))

def get_access_token_from_service_account():
    service_account_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    scopes = [
        "https://www.googleapis.com/auth/cloud-platform"
    ]
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=scopes
    )
    print("access_token before refresh:", credentials.token)
    request = Request()
    credentials.refresh(request)

    print("access_token: ", credentials.token)
    
    return credentials.token

def validate_ai_response(response: Dict) -> Dict:
    """Validate AI response structure"""
    validate(instance=response, schema=AI_RESPONSE_SCHEMA)
    return response

# === AI Analysis Function ===
def generate_ai_insights(query: str, data: List[Dict], analysis_type: str = "standard") -> Dict:
    """Generate AI analysis with type-specific prompts"""
    if not data:
        return {"error": "No data provided for analysis"}

    access_token = get_access_token_from_service_account()
    prompt_template = f"""
    [INST] <<SYS>>
    You are a data analyst. Provide {analysis_type} analysis.
    Focus only on the provided data. Format response as JSON:
    {{
        "key_findings": [],
        "potential_issues": [],
        "recommendations": [],
        "type_specific_insights": []
    }}
    <</SYS>>
    
    DATA SAMPLE: {json.dumps(data[:3], indent=2)}
    RECORD COUNT: {len(data)}
    [/INST]
    """

    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta/llama-3.1-405b-instruct-maas",
            "messages": [{"role": "user", "content": prompt_template}],
            "response_format": {"type": "json_object"},
            "temperature": 0.3 if analysis_type == "business" else 0.1,
            "max_tokens": 1024
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        print("AI response status code:", response.status_code)
        
        response.raise_for_status()
        response_json = response.json()
        
        # Get the content from the AI response
        ai_content = response_json['choices'][0]['message']['content']
        print("AI content (raw):", ai_content)
        
        # Parse the JSON string into a Python object
        try:
            parsed_content = json.loads(ai_content)
            print("Parsed AI content:", parsed_content)
        except json.JSONDecodeError as json_err:
            app.logger.error(f"Failed to parse AI response as JSON: {json_err}")
            return {"error": f"Invalid JSON response from AI: {json_err}"}
        
        # Transform the response to match expected schema
        transformed_content = transform_ai_response(parsed_content)
        print("Transformed AI content:", transformed_content)
        
        # Now validate the transformed object
        return validate_ai_response(transformed_content)

    except requests.exceptions.RequestException as req_err:
        app.logger.error(f"API request failed: {req_err}")
        return {"error": f"API request failed: {req_err}"}
    except ValidationError as val_err:
        app.logger.error(f"AI response validation failed: {val_err}")
        return {"error": f"AI response validation failed: {val_err}"}
    except Exception as e:
        app.logger.error(f"AI analysis failed: {str(e)}")
        return {"error": f"AI analysis failed: {str(e)}"}
    
def transform_ai_response(response: Dict) -> Dict:
    """Transform AI response to match expected schema format"""
    transformed = {
        "key_findings": [],
        "potential_issues": [],
        "recommendations": [],
        "type_specific_insights": []
    }
    
    # Process each field
    for field_name in transformed.keys():
        if field_name in response and isinstance(response[field_name], list):
            for item in response[field_name]:
                if isinstance(item, dict):
                    # Extract description if it exists
                    if "description" in item:
                        transformed[field_name].append(item["description"])
                    else:
                        # Convert entire dict to string as fallback
                        transformed[field_name].append(str(item))
                elif isinstance(item, str):
                    # Already a string, keep as is
                    transformed[field_name].append(item)
                else:
                    # Convert other types to string
                    transformed[field_name].append(str(item))
        elif field_name in response:
            # Handle non-list values
            if isinstance(response[field_name], str):
                transformed[field_name] = [response[field_name]]
            else:
                transformed[field_name] = [str(response[field_name])]
    
    return transformed


# === App Factory ===
app = Flask(__name__)
app.config.from_object(Config)

# Configure extensions
limiter.init_app(app)

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# === Preload cache ===
preload_registered_endpoints()

app.config['LLAMA_API_CONFIG'] = {
    'endpoint': API_URL,
    'max_tokens': 1024,
    'temperature': 0.3
}

@app.route('/generate-endpoint', methods=['POST'])
@limiter.limit("5 per minute")
@api_key_required
@api_error_handler
def generate_endpoint():
    """Generates a dynamic endpoint and stores the query in the database."""
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

    if endpoint_name in endpoint_registry:
        return jsonify({'Info': f'Endpoint already exists at {endpoint_registry[endpoint_name]['path']}'}), 400        

    query_str = utils.make_dynamic_query(schema_name, table_name, select_columns, filter_columns)
    created_at = utils.get_current_datetime()
    utils.insert_endpoint_to_db(endpoint_name, endpoint_path, query_str, created_by, created_at)

    # === Register endpoint in thread-safe dictionary ===
    with registry_lock:
        endpoint_registry[endpoint_name] = {
            "query": query_str,
            "path": endpoint_path,
            "created_at": created_at
        }

    return jsonify({
        'endpoint_path': endpoint_path,
        'method': 'GET',
        'ai_capabilities': {
            "available": True,
            "usage": "Add ?ai_analyze=true to request",
            "analysis_types": ["standard", "technical", "business"]
        }
    })

@app.route('/dynamic/<path:endpoint_name>', methods=['GET'])
@limiter.limit("60 per minute")
@api_key_required
@api_error_handler
def dynamic_endpoint(endpoint_name):
    """Handle dynamic endpoint requests using pre-registered endpoints"""
    # Check endpoint existence in registry (hashmap)
    if endpoint_name not in endpoint_registry:
        return jsonify({"error": f"Endpoint /{endpoint_name} not found"}), 404

    # Get query from registry
    endpoint_info = endpoint_registry[endpoint_name]
    query = endpoint_info["query"]  # This is already a string
    final_query = sql.SQL(query)
    
    if not query:
        return jsonify({'error': 'No query found for this endpoint'}), 404

    # Process filters/parameters
    filters = request.args.to_dict()
    non_sql_keys = ['ai_analyze', 'analysis_type']
    filtered_params = {k: v for k, v in filters.items() if k not in non_sql_keys}
    filter_values = list(filtered_params.values())

    # Execute query directly (no SQL composition needed)
    conn = psycopg2.connect(**utils.DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        cursor.execute(final_query, filter_values)
        columns = [desc[0] for desc in cursor.description]
        data = [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

    # Build response
    response = {
        "metadata": {
            "endpoint": endpoint_name,
            "query": query,  # Return the original query string
            "record_count": len(data)
        },
        "data": data
    }

    # Add AI analysis if requested
    if request.args.get('ai_analyze', '').lower() in ('true', '1', 'yes'):
        analysis_type = request.args.get('analysis_type', 'standard')
        response["ai_analysis"] = generate_ai_insights(query, data, analysis_type)

    return jsonify(response)


@app.route('/analyze/<analysis_type>/<endpoint_name>', methods=['GET'])
@limiter.limit("30 per minute")
@api_key_required
@api_error_handler
def analyze_endpoint(analysis_type, endpoint_name):
    """Dedicated analysis endpoints"""
    if analysis_type not in ('standard', 'business', 'technical'):
        return jsonify({"error": "Invalid analysis type"}), 400
        
    if endpoint_name not in endpoint_registry:
        return jsonify({"error": "Endpoint not found"}), 404


    # Execute query (same as dynamic endpoint)
    query = endpoint_registry[endpoint_name]["query"]
    params = {k: v for k, v in request.args.items() if k != 'verbose'}
    
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cursor = conn.cursor()
        cursor.execute(query, params)
        data = [dict(zip([col[0] for col in cursor.description], row)) 
               for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()


    return jsonify({
        "data": data,
        "analysis": generate_ai_insights(query, data, analysis_type),
        "metadata": {
            "analysis_type": analysis_type,
            "record_count": len(data)
        }
    })

# === Endpoint to list registered endpoints (optional) ===
@app.route('/list-endpoints', methods=['GET'])
@api_key_required
@api_error_handler
def list_registered_endpoints():
    with registry_lock:
        return jsonify({
            "registered_endpoints": list(endpoint_registry.keys()),
            "count": len(endpoint_registry)
        })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)