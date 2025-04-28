from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QSplitter, QApplication, QWidget, QVBoxLayout
import sys
from camera import CameraWidget
from login_form import LoginForm, UserInfoWidget
from control_button import ControlButtons
from status import StatusBar

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能人脸认证系统")
        self._init_ui()
        self._connect_signals()
        self.is_logged_in = False
        self.user_info = None  # 新增用户信息组件引用

    def _init_ui(self):
        # 创建组件
        self.camera = CameraWidget()
        self.login_form = LoginForm()
        self.controls = ControlButtons()
        self.status = StatusBar()
        self.current_method = "OpenCV"  # 默认检测方式
        self.status.update_detection_method(self.current_method)


        # 主布局
        main_splitter = QSplitter(Qt.Horizontal)
        left_splitter = QSplitter(Qt.Vertical)
        right_splitter = QSplitter(Qt.Vertical)

        # 左侧布局
        left_splitter.addWidget(self.camera)
        left_splitter.addWidget(self.status)
        left_splitter.setSizes([400, 200])


        # 右侧布局
        self.right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.login_form)
        right_layout.addWidget(self.controls)
        self.right_panel.setLayout(right_layout)
        right_splitter.setSizes([300, 200])

        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(self.right_panel)  # 替换原有right_splitter
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
        self.camera.camera_status_changed.connect(self.controls.update_camera_button)
        self.camera.camera_status_changed.connect(
            lambda on: self.status.show_message(
                f"摄像头已{'开启' if on else '关闭'}",
                is_error=not on
            )
        )

        # 状态更新
        self.camera.faces_detected.connect(self.status.update_face_count)
        self.camera.detection_toggled.connect(self.status.update_detection_status)
        # 检测状态变化消息
        self.camera.detection_toggled.connect(
            lambda detecting: self.status.show_message(
                f"人脸检测已{'开启' if detecting else '关闭'}",
                is_error=not detecting
            ) if self.camera._is_camera_on else None
        )

        # 摄像头关闭时自动关闭检测
        self.camera.camera_status_changed.connect(
            lambda on: self.camera.toggle_detection() if not on and self.camera._is_detecting else None
        )

        # 新增检测方式切换信号连接
        self.controls.method_changed.connect(self._on_method_changed)
        self.controls.method_changed.connect(self.camera.set_detection_method)

        # 登录控制
        self.login_form.login_clicked.connect(self._on_login)
        self.login_form.register_clicked.connect(self._on_register)
        self.controls.logout_clicked.connect(self._on_logout)

        # 新增FPS信号连接
        self.camera.fps_updated.connect(self.status.update_fps)

    def _on_method_changed(self, method):
        """处理检测方式切换"""
        self.current_method = method
        self.status.show_message(f"已切换至{method}检测模式")
        self.camera.blink_counter = 0
        self.status.update_detection_method(self.current_method)

    def _on_login(self, name, sid, pwd):
        self.is_logged_in = True

        # 创建用户信息组件
        self.user_info = UserInfoWidget(
            name=name,
            sid=sid,
            face_feature=None  # 暂时设为None，后续可从数据库获取
        )

        # 替换右侧面板内容
        self.right_panel.layout().replaceWidget(
            self.login_form, self.user_info
        )
        self.login_form.hide()
        self.user_info.show()

        # 更新状态
        self.controls.update_login_state(True)
        self.status.show_message(f"欢迎 {name}（{sid}）登录成功！")

    # GUI.py
    def _on_register(self, name, sid, pwd):
        # 基本输入验证
        if not all([name, sid, pwd]):
            self.status.show_message("注册失败：请填写完整信息", is_error=True)
            return

        # 摄像头状态检查
        if not self.camera._is_camera_on:
            self.status.show_message("注册失败：请先开启摄像头", is_error=True)
            return

        # 人脸检测检查
        if self.status.face_count.text() != "检测到人脸: 1":
            self.status.show_message("注册失败：需检测到单张人脸", is_error=True)
            return

        # 启用注册模式
        self.camera.registration_enabled = True

        # 获取检测数据
        if (self.camera.current_prob > 0.99 and
                self.camera.liveness_status and
                self.camera.current_face_feature is not None):

            # 创建用户信息字典
            user_data = {
                "name": name,
                "sid": sid,
                "password": pwd,
                "embedding": self.camera.current_face_feature[0].tolist()
            }

            # 控制台输出（后续可替换为数据库存储）
            print("新用户注册成功:")
            print(f"姓名: {user_data['name']}")
            print(f"学号: {user_data['sid']}")
            print(f"特征值样例: {user_data['embedding'][:5]}")

            # 更新用户信息显示
            if self.user_info:
                self.user_info.update_feature(user_data['embedding'])

            self.status.show_message(f"用户 {name} 注册成功")
        else:
            error_msg = "注册失败："
            if self.camera.current_prob <= 0.99:
                error_msg += f" 置信度不足({self.camera.current_prob:.2f})"
            if not self.camera.liveness_status:
                error_msg += " 活体检测未通过"
            self.status.show_message(error_msg, is_error=True)

        # 重置注册模式
        self.camera.registration_enabled = False

    def _on_logout(self):
        self.is_logged_in = False

        # 恢复登录表单
        self.right_panel.layout().replaceWidget(
            self.user_info, self.login_form
        )
        self.user_info.hide()
        self.login_form.show()

        # 更新状态
        self.controls.update_login_state(False)
        self.status.show_message("已安全退出登录")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
