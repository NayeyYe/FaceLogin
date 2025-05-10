import sys

sys.path.append("../")
import pymysql
from config import dbcfg
# 表结构定义SQL语句
TABLES = {
    'users': """
    CREATE TABLE IF NOT EXISTS users (
        student_id VARCHAR(20) PRIMARY KEY COMMENT '账号',
        name VARCHAR(50) NOT NULL COMMENT '姓名',
        password_hash VARCHAR(60) NOT NULL COMMENT 'bcrypt密码哈希',
        face_feature BLOB NOT NULL COMMENT '512维人脸特征向量',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
}

def check_database():
    """检查并创建数据库"""
    try:
        conn = pymysql.connect(
            host=dbcfg.host,
            port=dbcfg.port,
            user=dbcfg.user,
            password=dbcfg.password,
            charset='utf8mb4'
        )

        with conn.cursor() as cursor:
            # cursor.execute(
            #     f"DROP DATABASE IF NOT EXISTS {dbcfg.database} "
            # )
            # 创建数据库
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {dbcfg.database} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            # 切换数据库
            cursor.execute(f"USE {dbcfg.database}")

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
