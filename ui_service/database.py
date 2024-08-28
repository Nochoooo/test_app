import os
import psycopg2
from loguru import logger
from psycopg2 import DatabaseError

class Database:
    def __init__(self):
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            db_password = os.environ.get('POSTGRES_PASSWORD')
            db_host = os.environ.get('DB_HOST')
            db_port = os.environ.get('DB_PORT')
            db_name = os.environ.get('DB_NAME')
            db_user = os.environ.get('DB_USER')
            self.conn = psycopg2.connect(host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password)
            self.cursor = self.conn.cursor()
        except Exception as e:
            logger.error(f"Error retrieving DB connection parameters from environment: {str(e)}")
            return None

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def create_essays_table_if_not_exists(self):
        try:
            self.connect()
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS essays (
                    id SERIAL PRIMARY KEY,
                    author VARCHAR(100) NOT NULL,
                    content TEXT NOT NULL,
                    score REAL NOT NULL
                )
            """)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.close()

    def insert_data_into_essays(self, author, content, score):
        try:
            self.connect()
            self.cursor.execute("INSERT INTO essays (author, content, score) VALUES (%s, %s, %s)", (author, content, score))
            self.conn.commit()
            logger.info("Данные успешно вставлены в базу данных")
        except Exception as e:
            logger.error(f"Ошибка при вставке данных в базу данных: {e}")
            if self.conn:
                self.conn.rollback()
        finally:
            self.close()

    def get_all_essays(self):
        try:
            self.connect()
            self.cursor.execute("SELECT * FROM essays")
            essays = self.cursor.fetchall()
            return [{'id': essay[0], 'author': essay[1], 'content': essay[2], 'score': essay[3]} for essay in essays]
        except DatabaseError as db_error:
            logger.error(f"Database error during fetching essays: {db_error}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during fetching essays: {e}")
            return None
        finally:
            self.close()

    def delete_essays(self, ids):
        try:
            self.connect()
            self.cursor.execute("DELETE FROM essays WHERE id = ANY(%s)", (ids,))
            self.conn.commit()
        except DatabaseError as db_error:
            logger.error(f"Database error during deletion: {db_error}")
            if self.conn:
                self.conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Unexpected error during deletion: {e}")
            return False
        finally:
            self.close()
        return True

    def update_essay_score(self, essay_id, new_score):
        try:
            self.connect()
            self.cursor.execute("UPDATE essays SET score = %s WHERE id = %s", (new_score, essay_id))
            self.conn.commit()
        except DatabaseError as e:
            logger.error(f"Database error: {e}")
            self.conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False
        finally:
            self.close()
        return True
