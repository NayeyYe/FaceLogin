from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (QWidget, QFormLayout, QLineEdit,
                            QPushButton, QVBoxLayout, QLabel)
from PyQt5.QtGui import QFont

class LoginForm(QWidget):
    login_clicked = pyqtSignal(str, str, str)
    register_clicked = pyqtSignal(str, str, str)

    def __init__(self):
        super().__init__()
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("用户认证")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self._create_form(layout)
        self._create_buttons(layout)
        self.setLayout(layout)

    def _create_form(self, layout):
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(15)

        self.name_input = QLineEdit()
        self.id_input = QLineEdit()
        self.pwd_input = QLineEdit()
        self.pwd_input.setEchoMode(QLineEdit.Password)

        form_layout.addRow("姓名:", self.name_input)
        form_layout.addRow("学号:", self.id_input)
        form_layout.addRow("密码:", self.pwd_input)
        layout.addLayout(form_layout)

    def _create_buttons(self, layout):
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

    def _connect_signals(self):
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

class UserInfoWidget(QWidget):
    def __init__(self, name, sid, face_feature=None):
        super().__init__()
        self._init_ui(name, sid, face_feature)

    def _init_ui(self, name, sid, face_feature):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("用户信息")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self._create_info_form(layout, name, sid, face_feature)
        self.setLayout(layout)

    def _create_info_form(self, layout, name, sid, face_feature):
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(15)

        self.name_label = QLabel(name)
        self.sid_label = QLabel(sid)
        self.feature_label = QLabel(str(face_feature[:1] if face_feature else None))
        self.similarity_label = QLabel("")

        form_layout.addRow("姓名:", self.name_label)
        form_layout.addRow("学号:", self.sid_label)
        form_layout.addRow("特征值:", self.feature_label)
        form_layout.addRow("登录相似度:", self.similarity_label)

        style = "QLabel { font-size: 14px; padding: 5px; border-bottom: 1px solid #eee; }"
        for label in [self.name_label, self.sid_label, self.feature_label]:
            label.setStyleSheet(style)

        layout.addLayout(form_layout)

    def update_feature(self, new_feature):
        self.feature_label.setText(str(new_feature[:5]) if new_feature else None)

    def update_similarity(self, similarity):
        self.similarity_label.setText(f"{similarity:.2%}")
        color = "#4CAF50" if similarity >= 0.95 else "#FF5722"
        self.similarity_label.setStyleSheet(f"color: {color};")
