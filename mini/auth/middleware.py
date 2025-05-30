from functools import wraps
import psycopg2
import services.utils as utils
from flask import current_app
from flask import request, jsonify


def api_key_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        username = request.headers.get('X-Username')

        if not username or not api_key:
            return jsonify({"error": "Missing X-Username or X-API-Key in headers"}), 401
        
        if api_key != current_app.config.get('API_KEY'):
            return jsonify({"error": "Unauthorized"}), 401
        try:
            conn = psycopg2.connect(**utils.DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT username FROM de_dynamic_api.users
                WHERE username = %s
            """, (username,))
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            if not result or result[0] != username:
                return jsonify({"error": "Invalid username or API key"}), 401

        except Exception as e:
            current_app.logger.error(f"Auth DB error: {e}")
            return jsonify({"error": "Internal server error"}), 500
        
        return fn(*args, **kwargs)
    return wrapper
