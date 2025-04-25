from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QSplitter, QHBoxLayout, QWidget
from widgets.camera import CameraWidget
from widgets.control_button import ControlButtons
from widgets.login_form import LoginForm
from widgets.status import StatusBar
import sys
from PyQt5.QtWidgets import QApplication

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸识别系统")
        self._init_ui()

    def _init_ui(self):
        # 创建组件
        self.camera = CameraWidget()
        self.login_form = LoginForm()
        self.controls = ControlButtons()
        self.status = StatusBar()

        # 主布局
        main_splitter = QSplitter(Qt.Horizontal)

        # 左侧区域（摄像头 + 状态）
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(self.camera)
        left_splitter.addWidget(self.status)
        left_splitter.setSizes([600, 400])

        # 右侧区域（表单 + 控制）
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(self.login_form)
        right_splitter.addWidget(self.controls)
        right_splitter.setSizes([600, 400])

        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)

        # 窗口设置
        self.setCentralWidget(main_splitter)
        self.setMinimumSize(1280, 720)

        # 信号连接
        self._connect_signals()

    def _connect_signals(self):
        # 摄像头控制
        self.controls.camera_toggled.connect(self.camera.toggle_camera)
        self.camera.frame_updated.connect(
            lambda: self.status.update_camera_status(True)
        )
        self.camera.camera_status_changed.connect(self._handle_camera_status)
        self.camera.fps_updated.connect(self.status.update_fps)

        # 登录注册
        self.login_form.login_clicked.connect(self._on_login)
        self.login_form.register_clicked.connect(self._on_register)
        self.controls.logout_clicked.connect(self._on_logout)

    def _handle_camera_status(self, is_on):
        """统一处理摄像头状态变化"""
        self.controls.update_camera_button(is_on)
        self.status.update_camera_status(is_on)
        if not is_on:
            self.status.update_fps(0)


    def _on_login(self, name, pwd):
        # TODO: 连接后端验证
        self.status.show_message("登录成功！")
        self.controls.update_login_state(True)

    def _on_register(self, name, pwd):
        # TODO: 连接后端注册
        self.status.show_message("注册成功！")

    def _on_logout(self):
        self.controls.update_login_state(False)
        self.status.show_message("已退出登录")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())