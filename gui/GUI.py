import sys
sys.path.append("../")
import time
import traceback
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QSplitter, QApplication,
                             QWidget, QVBoxLayout)
from camera import CameraWidget
from login_form import LoginForm, UserInfoWidget
from control_button import ControlButtons
from status import StatusBar
from db.database_manager import DBOperator


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_logged_in = False
        self.user_info = {}
        self.registered_users = {}
        self.current_user = None
        self.current_method = "OpenCV"

        self._init_ui()
        self._connect_signals()
        self.status.update_detection_method(self.current_method)

    def _init_ui(self):
        self.camera = CameraWidget()
        self.login_form = LoginForm()
        self.controls = ControlButtons()
        self.status = StatusBar()

        main_splitter = QSplitter(Qt.Horizontal)
        left_splitter = QSplitter(Qt.Vertical)
        self.right_panel = QWidget()

        left_splitter.addWidget(self.camera)
        left_splitter.addWidget(self.status)
        left_splitter.setSizes([400, 200])

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.login_form)
        right_layout.addWidget(self.controls)
        self.right_panel.setLayout(right_layout)

        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(self.right_panel)
        main_splitter.setSizes([800, 400])

        self.setCentralWidget(main_splitter)
        self.setMinimumSize(1280, 720)
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QSplitter::handle { background-color: #ddd; }
        """)

    def _connect_signals(self):
        self.controls.camera_toggled.connect(self.camera.toggle_camera)
        self.controls.detection_toggled.connect(self.camera.toggle_detection)
        self.controls.logout_clicked.connect(self._on_logout)
        self.controls.method_changed.connect(self._on_method_changed)
        self.controls.method_changed.connect(self.camera.set_detection_method)

        self.camera.camera_status_changed.connect(self.status.update_camera_status)
        self.camera.camera_status_changed.connect(self.controls.update_camera_button)
        self.camera.faces_detected.connect(self.status.update_face_count)
        self.camera.detection_toggled.connect(self.status.update_detection_status)
        self.camera.fps_updated.connect(self.status.update_fps)

        self.login_form.login_clicked.connect(self._on_login)
        self.login_form.register_clicked.connect(self._on_register)

        self.camera.camera_status_changed.connect(
            lambda on: self.status.show_message(
                f"摄像头已{'开启' if on else '关闭'}", is_error=not on))

        self.camera.detection_toggled.connect(
            lambda detecting: self.status.show_message(
                f"人脸检测已{'开启' if detecting else '关闭'}", is_error=not detecting)
            if self.camera._is_camera_on else None)

        self.camera.camera_status_changed.connect(
            lambda on: self.camera.toggle_detection()
            if not on and self.camera._is_detecting else None)

    def _on_method_changed(self, method):
        self.current_method = method
        self.status.show_message(f"已切换至{method}检测模式")
        self.camera.blink_counter = 0
        self.status.update_detection_method(method)

    def _on_login(self, input_name, sid, pwd):
        try:
            if not all([input_name.strip(), sid.strip(), pwd]):
                self.status.show_message("登录失败：请填写完整信息", is_error=True)
                return

            if sid not in self.registered_users:
                self.status.show_message(f"学号 {sid} 未注册", is_error=True)
                return

            registered_data = self.registered_users[sid]
            if (input_name != registered_data["name"]) or (pwd != registered_data["password"]):
                self.status.show_message("姓名或密码错误", is_error=True)
                return

            if not self._check_camera_conditions():
                return

            if not (self.camera.current_prob > 0.99 and
                    self.camera.liveness_status and
                    self.camera.current_face_feature is not None):
                self._show_validation_errors()
                return

            similarity = self._calculate_similarity(registered_data)
            if similarity < 0.95:
                self.status.show_message(f"登录失败：相似度不足({similarity:.2f}<0.95)", is_error=True)
                return

            self._handle_successful_login(registered_data, sid, similarity)

        except Exception as e:
            self.status.show_message(f"登录异常：{str(e)}", is_error=True)
            print(f"登录出错：{traceback.format_exc()}")

    def _on_register(self, name, sid, pwd):
        if not all([name.strip(), sid.strip(), pwd]):
            self.status.show_message("注册失败：请填写完整信息", is_error=True)
            return

        if sid in self.registered_users:
            self.status.show_message(f"学号 {sid} 已被注册", is_error=True)
            return

        if not self._check_camera_conditions():
            return

        if not (self.camera.current_prob > 0.99 and
                self.camera.liveness_status and
                self.camera.current_face_feature is not None):
            self._show_validation_errors()
            return

        try:
            embedding = self.camera.current_face_feature[0].tolist()
            self.registered_users[sid] = {
                "name": name,
                "password": pwd,
                "embedding": embedding,
                "register_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            self.status.show_message(f"{name}({sid}) 注册成功！")
            self.login_form.name_input.setText(name)
            self.login_form.id_input.setText(sid)
            self.login_form.pwd_input.setText(pwd)

        except Exception as e:
            self.status.show_message(f"注册异常：{str(e)}", is_error=True)
            print(f"注册出错：{traceback.format_exc()}")
        finally:
            self.camera.registration_enabled = False

    def _on_logout(self):
        self.is_logged_in = False
        self.right_panel.layout().replaceWidget(self.user_info, self.login_form)
        self.user_info.hide()
        self.login_form.show()
        self.controls.update_login_state(False)
        self.status.show_message("已安全退出登录")

    def _check_camera_conditions(self):
        if not self.camera._is_camera_on:
            self.status.show_message("请先开启摄像头", is_error=True)
            return False
        if not self.camera._is_detecting:
            self.status.show_message("请开启人脸检测", is_error=True)
            return False
        if self.camera.detection_method == "OpenCV":
            self.status.show_message("请切换为MTCNN检测模式", is_error=True)
            return False
        if self.status.face_count.text() != "检测到人脸: 1":
            self.status.show_message("需检测到单张人脸", is_error=True)
            return False
        return True

    def _show_validation_errors(self):
        error_msg = "验证失败："
        conditions = []

        if self.camera.current_prob <= 0.99:
            conditions.append(f"置信度({self.camera.current_prob:.2f}<0.99)")
        if not self.camera.liveness_status:
            conditions.append("活体检测未通过")
        if self.camera.current_face_feature is None:
            conditions.append("未获取人脸特征")
        if abs(self.camera.current_angles[0]) < 85 or abs(self.camera.current_angles[1]) < 5 or abs(self.camera.current_angles[2]) < 5:
            conditions.append("未正对摄像头！")
        self.status.show_message(error_msg + " | ".join(conditions), is_error=True)

    def _calculate_similarity(self, registered_data):
        current_embedding = self.camera.current_face_feature[0]
        registered_embedding = np.array(registered_data["embedding"])
        return self.camera.face_system.calculate_similarity(
            current_embedding.reshape(1, -1),
            registered_embedding.reshape(1, -1)
        )[0][0]

    def _handle_successful_login(self, registered_data, sid, similarity):
        self.is_logged_in = True
        self.current_user = sid

        self.user_info = UserInfoWidget(
            name=registered_data["name"],
            sid=sid,
            face_feature=registered_data["embedding"])
        self.user_info.feature_label.setText(f"{registered_data['embedding'][:1]} ")
        self.user_info.similarity_label.setText(f"(相似度: {similarity:.2%})")

        self.right_panel.layout().replaceWidget(self.login_form, self.user_info)
        self.login_form.hide()
        self.user_info.show()

        self.controls.update_login_state(True)
        self.status.show_message(f"{registered_data['name']} 登录成功！相似度: {similarity:.2%}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
