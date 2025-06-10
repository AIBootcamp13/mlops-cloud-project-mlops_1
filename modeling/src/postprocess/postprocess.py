import os

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

def get_engine(db_name):
    try:
        user = os.environ.get('DB_USER')
        password = os.environ.get('DB_PASSWORD')
        host = os.environ.get('DB_HOST')
        port = os.environ.get('DB_PORT')

        if not all([user, password, host, port]):
            raise ValueError("One or more required DB environment variables are missing.")
        url = f"mysql+mysqldb://{user}:{password}@{host}:{port}/{db_name}"
        engine = create_engine(url)
        return engine

    except ValueError as ve:
        print(f"[ENV ERROR] {ve}")
    except SQLAlchemyError as e:
        print(f"[SQLAlchemy ERROR] Failed to create engine: {e}")
    except Exception as e:
        print(f"[GENERAL ERROR] Unexpected error: {e}")

def write_db(data: pd.DataFrame, db_name, table_name):
    engine = get_engine(db_name)

    connect = engine.connect()
    data.to_sql(table_name, connect, if_exists="append")
    connect.close()

def read_db(db_name, table_name, k=10):
    engine = get_engine(db_name)
    connect = engine.connect()
    result = connect.execute(
        statement=text(
            f"select recommend_content_id from {table_name} "
            f"order by `index` desc limit :k"
        ),
        parameters={"table_name": table_name, "k": k}
    )
    connect.close()
    contents = [data[0] for data in result]
    return contents