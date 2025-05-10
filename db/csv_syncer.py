# FaceLogin/db/csv_syncer.py
import sys
import os
import csv
import base64
import pymysql
from config import dbcfg


class CSVSyncer:
    def __init__(self):
        # 确保输出目录存在
        os.makedirs(os.path.dirname(dbcfg.csv), exist_ok=True)

    def _get_db_connection(self):
        """获取数据库连接"""
        return pymysql.connect(
            host=dbcfg.host,
            port=dbcfg.port,
            user=dbcfg.super_admin,
            password=dbcfg.password,
            database=dbcfg.database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    def _blob_to_base64(self, blob_data):
        """将BLOB数据转换为Base64字符串"""
        return base64.b64encode(blob_data).decode('utf-8') if blob_data else ''

    def sync_users_to_csv(self):
        """同步用户数据到CSV文件"""
        try:
            conn = self._get_db_connection()
            with conn:
                with conn.cursor() as cursor:
                    # 查询用户数据
                    cursor.execute(f"""
                        SELECT 
                            id, 
                            name, 
                            password_hash, 
                            face_feature,
                            created_at 
                        FROM {dbcfg.users_table}
                    """)
                    users = cursor.fetchall()

                # 写入CSV文件
                with open(dbcfg.csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # 写入表头
                    writer.writerow([
                        '用户ID',
                        '姓名',
                        '密码哈希',
                        '人脸特征(base64)',
                        '创建时间'
                    ])

                    # 写入数据
                    for user in users:
                        writer.writerow([
                            user['id'],
                            user['name'],
                            user['password_hash'],
                            self._blob_to_base64(user['face_feature']),
                            user['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                        ])

                print(f"成功同步 {len(users)} 条用户数据到 {dbcfg.csv}")
                return True

        except Exception as e:
            print(f"同步失败: {str(e)}")
            return False


if __name__ == "__main__":
    syncer = CSVSyncer()
    if syncer.sync_users_to_csv():
        sys.exit(0)
    else:
        sys.exit(1)
