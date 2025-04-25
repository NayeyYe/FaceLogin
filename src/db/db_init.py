import pymysql
from models import TABLES
from config import rootcfg

def check_database():
    """检查并创建数据库"""
    try:
        conn = pymysql.connect(
            host=rootcfg.host,
            port=rootcfg.port,
            user=rootcfg.user,
            password=rootcfg.password,
            charset='utf8mb4'
        )

        with conn.cursor() as cursor:
            # cursor.execute(
            #     f"DROP DATABASE IF NOT EXISTS {rootcfg.database} "
            # )
            # 创建数据库
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {rootcfg.database} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            # 切换数据库
            cursor.execute(f"USE {rootcfg.database}")

            # 创建数据表
            for table_name, ddl in TABLES.items():
                cursor.execute(ddl)
                print(f"成功创建表：{table_name}")

        conn.commit()
        print("数据库初始化完成！")

    except Exception as e:
        print(f"初始化失败: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    check_database()
