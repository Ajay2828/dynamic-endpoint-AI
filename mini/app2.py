import os
import threading
import json
import numpy as np
import pandas as pd
import hashlib
import tempfile
import atexit
import shutil
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import psycopg2
from psycopg2 import sql
from typing import Dict, List
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
from dotenv import load_dotenv
import requests
from datetime import datetime,timedelta,timezone
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from jsonschema import validate, ValidationError
from auth.middleware import api_key_required

# Load environment variables
load_dotenv()

# === Configuration ===
ENDPOINT = "us-central1-aiplatform.googleapis.com"
REGION = "us-central1"
PROJECT_ID = "saturam"
API_URL = f"https://{ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"

DB_HOST = os.getenv('DB_HOST', '34.58.9.128')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'postgres')
DB_USER = os.getenv('DB_USER', 'prod-shreemaruthi-private-db-user-sql')
DB_PASS = os.getenv('DB_PASS', 'CV-09r63p3n5H-KH')

DB_CONFIG = {
    'dbname': DB_NAME,
    'user': DB_USER,
    'password': DB_PASS,
    'port': DB_PORT,
    'host': DB_HOST
}

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

# === Utility Functions ===
def get_current_datetime():
    # Get the current local time
    local_now = datetime.now()
    # Define the IST offset (UTC+5:30)
    ist_offset = timedelta(hours=5, minutes=30)
    # Create the IST timezone
    ist_timezone = timezone(ist_offset)
    # Convert local time to IST
    ist_now = local_now.astimezone(ist_timezone)
    return ist_now.strftime('%Y-%m-%d %H:%M:%S')

def get_table_schema(schema_name: str, table_name: str) -> Dict[str, str]:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = %s AND table_name = %s
        """, (schema_name, table_name))
    schema = {row[0]: row[1] for row in cursor.fetchall()}
    cursor.close()
    conn.close()
    return schema

def generate_endpoint_name(table_name: str, select_columns: List[str], filter_columns: List[str]) -> str:
    select_part = "_".join(select_columns)
    filter_part = "_".join(filter_columns) if filter_columns else "nofilter"
    endpoint_name = f"{table_name}__select_{select_part}__filter_{filter_part}"
    endpoint_name = endpoint_name.lower().replace(" ", "_")
    return endpoint_name

def make_dynamic_query(schema_name: str, table_name: str, select_columns: List[str], filter_columns: List[str]) -> str:
    """
    Returns a parameterized SQL query string with %s placeholders for filters.
    This function does NOT execute the query — just returns the query as a string.
    """
    select_clause = sql.SQL(', ').join(map(sql.Identifier, select_columns))

    base_query = sql.SQL("SELECT {} FROM {}.{}").format(
        select_clause,
        sql.Identifier(schema_name),
        sql.Identifier(table_name)
    )

    # Create WHERE clause with placeholders
    where_clauses = [
        sql.SQL("{} = %s").format(sql.Identifier(col)) for col in filter_columns
    ]

    if where_clauses:
        query = sql.SQL("{} WHERE {}").format(
            base_query,
            sql.SQL(" AND ").join(where_clauses)
        )
    else:
        query = base_query

    # Return the final query as a string (with %s placeholders)
    return query.as_string(psycopg2.connect(**DB_CONFIG))

def insert_endpoint_to_db(endpoint_name, endpoint_path, query, created_by, created_at):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO de_dynamic_api.endpoint_registry (endpoint_name, endpoint_path, query, created_by, created_at) VALUES (%s, %s, %s, %s, %s)",
            (endpoint_name, endpoint_path, query, created_by, created_at)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

# === ML Components ===
class DataHistoryManager:
    """Manages temporary storage of historical data"""
    def __init__(self, max_records=5000):
        self.max_records = max_records
        self.temp_dir = tempfile.mkdtemp(prefix="api_ml_")
        self.endpoint_hashes = {}
        atexit.register(self.cleanup)
        
    def _get_endpoint_hash(self, endpoint_name):
        return hashlib.md5(endpoint_name.encode()).hexdigest()
    
    def get_history_path(self, endpoint_name):
        return f"{self.temp_dir}/{self._get_endpoint_hash(endpoint_name)}.parquet"
    
    def update_history(self, endpoint_name, new_data):
        """Maintain rolling history in parquet files"""
        path = self.get_history_path(endpoint_name)
        new_df = pd.DataFrame(new_data)
        
        try:
            history_df = pd.read_parquet(path)
            combined = pd.concat([history_df, new_df])
            combined = combined.tail(self.max_records)
        except (FileNotFoundError, OSError):
            combined = new_df.tail(self.max_records)
        
        combined.to_parquet(path)
        return combined
    
    def cleanup(self):
        """Clean up temporary files on exit"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

class MLInsightGenerator:
    """Generates ML insights from historical data"""
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.data_manager = DataHistoryManager(max_records=5000)
        
    def _detect_data_types(self, df):
        """Analyze dataframe columns and return typed information"""
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        date_cols = [col for col in df.columns if 
                    df[col].dtype == 'datetime64[ns]' or 
                    'date' in col.lower() or 
                    'time' in col.lower()]
        
        return {
            "numeric": num_cols,
            "text": text_cols,
            "date": date_cols
        }
    
    def generate_insights(self, endpoint_name, current_data):
        """Main ML analysis workflow"""
        full_data = self.data_manager.update_history(endpoint_name, current_data)
        data_types = self._detect_data_types(full_data)
        
        insights = {
            "data_characteristics": {
                "total_records": len(full_data),
                "columns_analyzed": data_types
            }
        }
        
        # Numeric Analysis
        if data_types["numeric"]:
            numeric_insights = self._analyze_numeric(full_data[data_types["numeric"]])
            insights.update(numeric_insights)
        
        # Text Analysis
        if data_types["text"]:
            text_insights = self._analyze_text(full_data[data_types["text"]])
            insights.update(text_insights)
        
        # Temporal Analysis
        if data_types["date"] and len(full_data) > 30:
            temporal_insights = self._analyze_temporal(full_data, data_types["date"])
            insights.update(temporal_insights)
        
        return insights
    
    def _analyze_numeric(self, numeric_df):
        """Analyze numeric columns"""
        insights = {}
        
        # Anomaly Detection
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomalies = iso_forest.fit_predict(numeric_df)
        insights["anomaly_detection"] = {
            "algorithm": "Isolation Forest",
            "anomaly_count": int((anomalies == -1).sum()),
            "anomaly_percentage": float((anomalies == -1).mean())
        }
        
        # Clustering
        if len(numeric_df.columns) > 1 and len(numeric_df) > 10:
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(numeric_df)
            
            k = min(3, len(numeric_df))
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(reduced)
            
            insights["numeric_clusters"] = {
                "algorithm": "PCA + KMeans",
                "cluster_counts": pd.Series(clusters).value_counts().to_dict(),
                "explained_variance": pca.explained_variance_ratio_.tolist()
            }
        
        return {"numeric_insights": insights}
    
    def _analyze_text(self, text_df):
        """Analyze text columns"""
        insights = {}
        
        for col in text_df.columns:
            texts = text_df[col].dropna().astype(str).tolist()
            if len(texts) < 5:
                continue
                
            embeddings = self.sentence_model.encode(texts)
            k = min(3, len(texts))
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            cluster_info = {}
            for cluster_id in range(k):
                cluster_texts = np.array(texts)[clusters == cluster_id]
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(
                    embeddings[clusters == cluster_id] - centroid, 
                    axis=1
                )
                representative = cluster_texts[np.argmin(distances)]
                
                cluster_info[f"cluster_{cluster_id}"] = {
                    "count": len(cluster_texts),
                    "representative_text": representative,
                    "sample_texts": cluster_texts[:3].tolist()
                }
            
            insights[col] = cluster_info
        
        return {"text_insights": insights} if insights else {}
    
    def _analyze_temporal(self, df, date_cols):
        """Analyze temporal patterns"""
        insights = {}
        primary_date_col = date_cols[0]
        
        df['_temp_date'] = pd.to_datetime(df[primary_date_col])
        df = df.sort_values('_temp_date')
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            window_size = min(7, len(df))
            rolling_stats = df.set_index('_temp_date')[numeric_cols].rolling(window_size).agg(['mean', 'std'])
            
            insights["temporal_trends"] = {
                "window_size": window_size,
                "recent_trends": rolling_stats.iloc[-window_size:].to_dict()
            }
        
        return {"temporal_insights": insights} if insights else {}

# === AI Analysis Functions ===
def get_access_token_from_service_account():
    service_account_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=scopes
    )
    request = Request()
    credentials.refresh(request)
    return credentials.token

def generate_ai_insights(query: str, data: List[Dict], analysis_type: str = "standard", custom_prompt: str = None) -> Dict:
    """Generate AI analysis with optional ML-enhanced prompts"""
    if not data:
        return {"error": "No data provided for analysis"}

    access_token = get_access_token_from_service_account()
    prompt_template = custom_prompt or f"""
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
        response.raise_for_status()
        response_json = response.json()
        ai_content = response_json['choices'][0]['message']['content']
        
        try:
            parsed_content = json.loads(ai_content)
            transformed_content = transform_ai_response(parsed_content)
            return validate_ai_response(transformed_content)
        except json.JSONDecodeError as json_err:
            return {"error": f"Invalid JSON response from AI: {json_err}"}

    except requests.exceptions.RequestException as req_err:
        return {"error": f"API request failed: {req_err}"}
    except Exception as e:
        return {"error": f"AI analysis failed: {str(e)}"}

def transform_ai_response(response: Dict) -> Dict:
    """Transform AI response to match expected schema"""
    transformed = {
        "key_findings": [],
        "potential_issues": [],
        "recommendations": [],
        "type_specific_insights": []
    }
    
    for field_name in transformed.keys():
        if field_name in response and isinstance(response[field_name], list):
            for item in response[field_name]:
                if isinstance(item, dict):
                    transformed[field_name].append(item.get("description", str(item)))
                elif isinstance(item, str):
                    transformed[field_name].append(item)
                else:
                    transformed[field_name].append(str(item))
        elif field_name in response:
            if isinstance(response[field_name], str):
                transformed[field_name] = [response[field_name]]
            else:
                transformed[field_name] = [str(response[field_name])]
    
    return transformed

def validate_ai_response(response: Dict) -> Dict:
    """Validate AI response structure"""
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
    validate(instance=response, schema=AI_RESPONSE_SCHEMA)
    return response

# === Flask Application ===
app = Flask(__name__)
limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)

# Initialize ML and registry components
ml_generator = MLInsightGenerator()
endpoint_registry = {}
registry_lock = threading.Lock()

def preload_registered_endpoints():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT endpoint_name, query, endpoint_path FROM de_dynamic_api.endpoint_registry")
        rows = cursor.fetchall()
        with registry_lock:
            for name, query, path in rows:
                endpoint_registry[name] = {"query": query, "path": path}
        cursor.close()
        conn.close()
    except Exception as e:
        print("Failed to preload endpoints:", str(e))

# Preload endpoints on startup
preload_registered_endpoints()
print(f"Loaded {len(endpoint_registry)} endpoints from database.")
print(endpoint_registry)

# === API Endpoints ===
@app.route('/generate-endpoint', methods=['POST'])
@limiter.limit("5 per minute")
# @api_key_required
@api_error_handler
def generate_endpoint():
    """Generate a new dynamic endpoint"""
    data = request.get_json()
    
    table_name = data.get('table_name')
    schema_name = data.get('schema_name', 'public')
    filter_columns = data.get('filter_columns', [])
    select_columns = data.get('select_columns', [])
    created_by = data.get('created_by', 'anonymous')

    if not table_name or not select_columns:
        return jsonify({'error': 'Both table_name and select_columns are required'}), 400
    
    schema = get_table_schema(schema_name, table_name)
    if not schema:
        return jsonify({'error': f'Table {table_name} not found'}), 404

    for col in select_columns + filter_columns:
        if col not in schema:
            return jsonify({'error': f'Column {col} not found in table {table_name}'}), 400

    endpoint_name = generate_endpoint_name(table_name, select_columns, filter_columns)
    endpoint_path = f'/dynamic/{endpoint_name}'

    if endpoint_name in endpoint_registry:
        return jsonify({'info': f'Endpoint already exists at {endpoint_registry[endpoint_name]["path"]}'}), 400        

    query_str = make_dynamic_query(schema_name, table_name, select_columns, filter_columns)
    created_at = get_current_datetime()
    insert_endpoint_to_db(endpoint_name, endpoint_path, query_str, created_by, created_at)

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
        },
        'ml_capabilities': {
            "available": True,
            "usage": "Add ?ml_analyze=true to request",
            "history_size": 5000
        }
    })

@app.route('/dynamic/<path:endpoint_name>', methods=['GET'])
@limiter.limit("60 per minute")
# @api_key_required
@api_error_handler
def dynamic_endpoint(endpoint_name):
    """Handle dynamic endpoint requests with ML/AI options"""
    if endpoint_name not in endpoint_registry:
        return jsonify({"error": f"Endpoint /{endpoint_name} not found"}), 404
    

    # Process filters/parameters
    filters = request.args.to_dict()
    non_sql_keys = ['ai_analyze', 'analysis_type','ml_analyze']
    filtered_params = {k: v for k, v in filters.items() if k not in non_sql_keys}
    filter_values = list(filtered_params.values())

    query = endpoint_registry[endpoint_name]["query"]
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql.SQL(query), filter_values)
        columns = [desc[0] for desc in cursor.description]
        data = [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

    response = {
        "metadata": {
            "endpoint": endpoint_name,
            "record_count": len(data)
        },
        "data": data
    }

    # ML Analysis
    if request.args.get('ml_analyze', '').lower() in ('true', '1', 'yes'):
        ml_insights = ml_generator.generate_insights(endpoint_name, data)
        response["ml_insights"] = ml_insights
        
        # Enhanced AI analysis with ML context
        if request.args.get('ai_analyze', '').lower() in ('true', '1', 'yes'):
            analysis_type = request.args.get('analysis_type', 'standard')
            enhanced_prompt = (
                "ML ANALYSIS SUMMARY:\n"
                f"{json.dumps(ml_insights, indent=2)}\n\n"
                f"USER REQUEST: Provide {analysis_type} analysis incorporating these ML insights."
            )
            response["ai_analysis"] = generate_ai_insights(
                query, 
                data, 
                analysis_type,
                custom_prompt=enhanced_prompt
            )
    elif request.args.get('ai_analyze', '').lower() in ('true', '1', 'yes'):
        # Standard AI analysis
        analysis_type = request.args.get('analysis_type', 'standard')
        response["ai_analysis"] = generate_ai_insights(query, data, analysis_type)

    return jsonify(response)

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
