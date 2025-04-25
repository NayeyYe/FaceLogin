# 表结构定义SQL语句
TABLES = {
    'users': """
    CREATE TABLE IF NOT EXISTS users (
        student_id VARCHAR(20) PRIMARY KEY COMMENT '账号',
        name VARCHAR(50) NOT NULL COMMENT '姓名',
        gender ENUM('M', 'F', 'O') COMMENT '性别枚举',
        password_hash VARCHAR(60) NOT NULL COMMENT 'bcrypt密码哈希',
        face_feature BLOB NOT NULL COMMENT '512维人脸特征向量',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    'admins': """
    CREATE TABLE IF NOT EXISTS admins (
        admin_id INT AUTO_INCREMENT PRIMARY KEY,
        role ENUM('admin', 'superadmin') NOT NULL,
        permissions TEXT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
}
