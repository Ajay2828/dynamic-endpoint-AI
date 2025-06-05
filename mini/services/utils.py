import psycopg2
from psycopg2 import sql
from typing import Dict, List
from datetime import datetime,timedelta,timezone
from config import DB_CONFIG


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

def generate_endpoint_name_1(table_name: str) -> str:
    """Generate unique endpoint name"""
    return hashlib.md5(table_name.encode()).hexdigest()[:8]


def generate_endpoint_name(table_name: str, select_columns: List[str], filter_columns: List[str]) -> str:
    select_part = "_".join(select_columns)
    filter_part = "_".join(filter_columns) if filter_columns else "nofilter"
    endpoint_name = f"{table_name}__select_{select_part}__filter_{filter_part}"
    endpoint_name = endpoint_name.lower().replace(" ", "_")
    return endpoint_name


def insert_endpoint_to_db(endpoint_name, endpoint_path, query_str, created_by, created_at):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO de_dynamic_api.endpoint_registry (endpoint_name, endpoint_path, query, created_by, created_at)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
    """, (endpoint_name, endpoint_path, query_str, created_by, created_at))
    endpoint_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    return endpoint_id

def endpoint_exists(endpoint_name: str) -> bool:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM de_dynamic_api.endpoint_registry WHERE endpoint_name = %s LIMIT 1;",
        (endpoint_name,)
    )
    exists = cursor.fetchone() is not None
    cursor.close()
    conn.close()
    return exists

def get_query_by_endpoint_name(endpoint_name: str):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, query FROM de_dynamic_api.endpoint_registry WHERE endpoint_name = %s
    """, (endpoint_name,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    if result:
        return {'id': result[0], 'query': result[1]}
    else:
        return None


def log_endpoint_usage(endpoint_id: int, accessed_at: str):
    """ Logs the usage of a dynamic endpoint in the database. """
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO de_dynamic_api.endpoint_usage (endpoint_id, accessed_at)
        VALUES (%s, %s)
    """, (endpoint_id, accessed_at))
    conn.commit()
    cursor.close()
    conn.close()