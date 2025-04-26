from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QSplitter, QApplication
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能人脸认证系统")
        self._init_ui()
        self._connect_signals()
        self.is_logged_in = False

    def _init_ui(self):
        # 创建组件
        from widgets.camera import CameraWidget
        from widgets.login_form import LoginForm
        from widgets.control_button import ControlButtons
        from widgets.status import StatusBar

        self.camera = CameraWidget()
        self.login_form = LoginForm()
        self.controls = ControlButtons()
        self.status = StatusBar()

        # 主布局
        main_splitter = QSplitter(Qt.Horizontal)
        left_splitter = QSplitter(Qt.Vertical)
        right_splitter = QSplitter(Qt.Vertical)

        # 左侧布局
        left_splitter.addWidget(self.camera)
        left_splitter.addWidget(self.status)
        left_splitter.setSizes([400, 200])

        # 右侧布局
        right_splitter.addWidget(self.login_form)
        right_splitter.addWidget(self.controls)
        right_splitter.setSizes([300, 200])

        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([800, 400])

        self.setCentralWidget(main_splitter)
        self.setMinimumSize(1280, 720)
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QSplitter::handle { background-color: #ddd; }
        """)

    def _connect_signals(self):
        # 摄像头控制
        self.controls.camera_toggled.connect(self.camera.toggle_camera)
        self.camera.camera_status_changed.connect(self.status.update_camera_status)
        self.controls.detection_toggled.connect(self.camera.toggle_detection)

        # 状态更新
        self.camera.faces_detected.connect(self.status.update_face_count)
        self.camera.detection_toggled.connect(self.status.update_detection_status)

        # 登录控制
        self.login_form.login_clicked.connect(self._on_login)
        self.login_form.register_clicked.connect(self._on_register)
        self.controls.logout_clicked.connect(self._on_logout)


    def _on_login(self, name, sid, pwd):
        # TODO: 连接实际认证逻辑
        self.is_logged_in = True
        self.controls.update_login_state(True)
        self.status.show_message(f"欢迎 {name}（{sid}）登录成功！")
        self.login_form.setVisible(False)

    def _on_register(self, name, sid, pwd):
        # TODO: 连接实际注册逻辑
        self.status.show_message(f"用户 {name}（{sid}）注册成功")

    def _on_logout(self):
        self.is_logged_in = False
        self.controls.update_login_state(False)
        self.login_form.setVisible(True)
        self.status.show_message("已安全退出登录")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
