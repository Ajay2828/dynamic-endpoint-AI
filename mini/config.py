import os
from dotenv import load_dotenv


load_dotenv()
# PostgreSQL
# DB_HOST = os.getenv('DB_HOST', '34.58.9.128')
# DB_PORT = os.getenv('DB_PORT', '5432')
# DB_NAME = os.getenv('DB_NAME', 'postgres')
# DB_USER = os.getenv('DB_USER', 'shreemaruti_readonly')
# DB_PASS = os.getenv('DB_PASS', 'G7c!bX9@q12')

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

class Config:
    # Key Configuration
    API_KEY = os.getenv('API_KEY', 'b9c1e0e17a90f284b89b2e8b34b8d04eeaacec52025deeefc494a189e5ad70a3')
    
    # Rate Limiting
    RATE_LIMIT = os.getenv('RATE_LIMIT', '5/minute')