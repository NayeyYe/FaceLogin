import pymysql
from pymysql import cursors
from pymysql.constants import CLIENT
from config import dbcfg


class MySQLManager:
    """MySQL连接管理器（连接池模式）"""

    def __init__(self):
        self.pool = pymysql.pool.ConnectionPool(
            min_size=3,
            max_size=10,
            host=dbcfg.host,
            port=dbcfg.port,
            user=dbcfg.user,
            password=dbcfg.password,
            database=dbcfg.database,
            charset='utf8mb4',
            client_flag=CLIENT.MULTI_STATEMENTS,
            cursorclass=cursors.DictCursor
        )

    def get_connection(self):
        """获取数据库连接"""
        return self.pool.connection()

    def execute_query(self, sql, params=None):
        """执行查询语句（返回结果集）"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params or ())
                return cursor.fetchall()

    def execute_update(self, sql, params=None):
        """执行更新语句（返回影响行数）"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                rows = cursor.execute(sql, params or ())
                conn.commit()
                return rows


# 全局数据库管理器实例
db_manager = MySQLManager()
