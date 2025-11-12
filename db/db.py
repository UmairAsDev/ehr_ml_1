
from sqlalchemy import create_engine, text, URL
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv


load_dotenv()

db_host = os.getenv("DB_LOCAL_HOST", "localhost")
db_port = os.getenv("DB_LOCAL_PORT", "3306")
db_name = os.getenv("DB_LOCAL_NAME", "test_db")
db_username = os.getenv("DB_LOCAL_USERNAME", "root")
db_password = os.getenv("DB_LOCAL_PASSWORD", "")

url_object = URL.create(
    "mysql+pymysql",
    username=db_username,
    password=db_password, 
    host=db_host,
    port=3306,
    database=db_name,
)

print("database url", url_object)



engine = create_engine(url_object, pool_pre_ping=True)


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)






def get_db():
    db = SessionLocal()
    try: 
        yield db
    finally:
        db.close()
    