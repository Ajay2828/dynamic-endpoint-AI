from flask import request, jsonify
import psycopg2
from psycopg2 import sql
from app import app
import services.utils as utils
from app import limiter
from auth.middleware import api_key_required
import traceback



@app.route('/generate-endpoint', methods=['POST'])
@limiter.limit(app.config['RATE_LIMIT'])
@api_key_required
def generate_endpoint():
    """ Generates a dynamic endpoint and a dynamic query will be stored in the database."""
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

        # Validate columns exist in table
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

        return jsonify({
            'message': f'Endpoint created at {endpoint_path}',
            'endpoint_path': endpoint_path,
            'method': 'GET'
        })
    
    except Exception as e:
        app.logger.error(traceback.format_exc())
        app.logger.error('Dynamic endpoint Generation API Error: {}'.format(e))
        return jsonify({"error": "Internal server error"}), 500


# Catch-all route for dynamic endpoints
@app.route('/dynamic/<path:endpoint_name>', methods=['GET'])
@limiter.limit(app.config['RATE_LIMIT'])
@api_key_required
def handle_dynamic_endpoint(endpoint_name):
    try:
        if not utils.endpoint_exists(endpoint_name):
            return jsonify({'error': f'Endpoint /{endpoint_name} not found'}), 404
        
        filters = request.args.to_dict()

        filter_values = list(filters.values()) 
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
        cursor.close()
        conn.close()

        utils.log_endpoint_usage(endpoint_id, accessed_at=utils.get_current_datetime())
        
        return jsonify([dict(zip(columns, rows))]), 200
    
    except Exception as e:
        app.logger.error(traceback.format_exc())
        app.logger.logging.error('Dynamic API Error: {}'.format(e))
        return jsonify({"error": "Internal server error"}), 500
    
@app.route('/dynamic/list', methods=['GET'])
@api_key_required
def list_dynamic_endpoints():
    try:
        conn = psycopg2.connect(**utils.DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, endpoint_name, endpoint_path, created_by, created_at 
            FROM de_dynamic_api.endpoint_registry 
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()

        return jsonify([dict(zip(columns, row)) for row in rows]), 200

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to fetch dynamic endpoints'}), 500
