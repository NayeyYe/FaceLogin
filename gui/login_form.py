from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (QWidget, QFormLayout, QLineEdit,
                            QPushButton, QVBoxLayout, QLabel)
from PyQt5.QtGui import QFont

class LoginForm(QWidget):
    login_clicked = pyqtSignal(str, str, str)  # 新增student_id
    register_clicked = pyqtSignal(str, str, str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title = QLabel("用户认证")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 表单布局
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(15)

        self.name_input = QLineEdit()
        self.id_input = QLineEdit()  # 新增学号输入
        self.pwd_input = QLineEdit()
        self.pwd_input.setEchoMode(QLineEdit.Password)

        form_layout.addRow("姓名:", self.name_input)
        form_layout.addRow("学号:", self.id_input)
        form_layout.addRow("密码:", self.pwd_input)
        layout.addLayout(form_layout)

        # 按钮
        btn_style = """
            QPushButton {
                padding: 8px; 
                font-size: 14px;
                background-color: #4CAF50;
                color: white;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #45a049; }
        """

        self.login_btn = QPushButton("立即登录")
        self.register_btn = QPushButton("注册新用户")
        self.login_btn.setStyleSheet(btn_style)
        self.register_btn.setStyleSheet(btn_style.replace("#4CAF50", "#2196F3"))

        layout.addWidget(self.login_btn)
        layout.addWidget(self.register_btn)
        self.setLayout(layout)

        # 信号连接
        self.login_btn.clicked.connect(self._on_login)
        self.register_btn.clicked.connect(self._on_register)

    def _on_login(self):
        self.login_clicked.emit(
            self.name_input.text(),
            self.id_input.text(),
            self.pwd_input.text()
        )

    def _on_register(self):
        self.register_clicked.emit(
            self.name_input.text(),
            self.id_input.text(),
            self.pwd_input.text()
        )
