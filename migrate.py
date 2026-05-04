import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.environ.get("NEON_DATABASE_URL"))
with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE documents ADD COLUMN user_id VARCHAR;"))
        conn.commit()
        print("Column added successfully")
    except Exception as e:
        print(f"Error: {e}")
