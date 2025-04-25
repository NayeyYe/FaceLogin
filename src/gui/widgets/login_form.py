from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QFormLayout, QLineEdit, QPushButton, QVBoxLayout


class LoginForm(QWidget):
    login_clicked = pyqtSignal(str, str)
    register_clicked = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # 输入框
        self.name_input = QLineEdit()
        self.pwd_input = QLineEdit()
        self.pwd_input.setEchoMode(QLineEdit.Password)

        form_layout = QFormLayout()
        form_layout.addRow("姓名:", self.name_input)
        form_layout.addRow("密码:", self.pwd_input)
        layout.addLayout(form_layout)

        # 按钮
        self.login_btn = QPushButton("登录")
        self.register_btn = QPushButton("注册")
        self.login_btn.clicked.connect(self._on_login)
        self.register_btn.clicked.connect(self._on_register)

        layout.addWidget(self.login_btn)
        layout.addWidget(self.register_btn)
        self.setLayout(layout)

    def _on_login(self):
        self.login_clicked.emit(
            self.name_input.text(),
            self.pwd_input.text()
        )

    def _on_register(self):
        self.register_clicked.emit(
            self.name_input.text(),
            self.pwd_input.text()
        )
