import time
import traceback

import numpy as np
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
        self.user_info = {}  # 新增用户信息组件引用
        self.registered_users = {}  # 更改变量名避免冲突
        self.current_user = None  # 当前登录用户

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

    # 在GUI.py的MainWindow类中修改_on_login方法
    def _on_login(self, input_name, sid, pwd):
        """增强版登录逻辑"""
        try:
            # 基础验证
            if not all([input_name.strip(), sid.strip(), pwd]):
                self.status.show_message("登录失败：请填写完整信息", is_error=True)
                return

            # 检查学号是否存在
            if sid not in self.registered_users:
                self.status.show_message(f"学号 {sid} 未注册", is_error=True)
                return

            # 获取注册信息
            registered_data = self.registered_users[sid]

            # 验证姓名和密码
            if (input_name != registered_data["name"]) or (pwd != registered_data["password"]):
                self.status.show_message("姓名或密码错误", is_error=True)
                return

            # 摄像头状态检查
            if not self.camera._is_camera_on:
                self.status.show_message("登录失败：请先开启摄像头", is_error=True)
                return

            if not self.camera._is_detecting:
                self.status.show_message("登录失败：请开启人脸检测", is_error=True)
                return

            if self.camera.detection_method == "OpenCV":
                self.status.show_message("请切换为MTCNN检测模式", is_error=True)
                return

            # 人脸检测检查
            if self.status.face_count.text() != "检测到人脸: 1":
                self.status.show_message("登录失败：需检测到单张人脸", is_error=True)
                return

            # 生物特征验证
            if not (self.camera.current_prob > 0.99 and
                    self.camera.liveness_status and
                    self.camera.current_face_feature is not None):
                error_msg = "登录失败："
                conditions = []
                if self.camera.current_prob <= 0.99:
                    conditions.append(f"置信度({self.camera.current_prob:.2f}<0.99)")
                if not self.camera.liveness_status:
                    conditions.append("活体检测未通过")
                if self.camera.current_face_feature is None:
                    conditions.append("未获取人脸特征")
                self.status.show_message(error_msg + " | ".join(conditions), is_error=True)
                return

            # 计算相似度
            current_embedding = self.camera.current_face_feature[0]
            registered_embedding = np.array(registered_data["embedding"])

            # 使用MTCNN的相似度计算方法
            similarity = self.camera.face_system.calculate_similarity(
                current_embedding.reshape(1, -1),
                registered_embedding.reshape(1, -1)
            )[0][0]

            # 相似度验证
            if similarity < 0.95:
                self.status.show_message(f"登录失败：相似度不足({similarity:.2f}<0.95)", is_error=True)
                return

            # 登录成功处理
            self.is_logged_in = True
            self.current_user = sid

            # 更新用户信息显示
            self.user_info = UserInfoWidget(
                name=registered_data["name"],
                sid=sid,
                face_feature=registered_data["embedding"]
            )
            self.user_info.feature_label.setText(
                f"{registered_data['embedding'][:1]} "
            )
            self.user_info.similarity_label.setText(
                f"(相似度: {similarity:.2%})"

            )

            # 替换右侧面板
            self.right_panel.layout().replaceWidget(
                self.login_form, self.user_info
            )
            self.login_form.hide()
            self.user_info.show()

            # 控制台输出
            print("\n登录成功信息：")
            print(f"学号: {sid}")
            print(f"姓名: {registered_data['name']}")
            print(f"相似度: {similarity:.2%}")
            print(f"登录时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            # 状态更新
            self.controls.update_login_state(True)
            self.status.show_message(f"{input_name} 登录成功！相似度: {similarity:.2%}")

        except Exception as e:
            self.status.show_message(f"登录异常：{str(e)}", is_error=True)
            print(f"登录出错：{traceback.format_exc()}")

    def _on_register(self, name, sid, pwd):
        """增强版注册逻辑"""
        # 输入验证
        if not all([name.strip(), sid.strip(), pwd]):
            self.status.show_message("注册失败：请填写完整信息", is_error=True)
            return

        # 检查学号是否已存在
        if sid in self.registered_users:
            self.status.show_message(f"学号 {sid} 已被注册", is_error=True)
            return

        # 摄像头状态检查
        if not self.camera._is_camera_on:
            self.status.show_message("注册失败：请先开启摄像头", is_error=True)
            return

        if not self.camera._is_detecting:
            self.status.show_message("登录失败：请开启人脸检测", is_error=True)
            return

        if self.camera.detection_method == "OpenCV":
            self.status.show_message("请切换为MTCNN检测模式", is_error=True)
            return

        # 人脸检测检查
        if self.status.face_count.text() != "检测到人脸: 1":
            self.status.show_message("注册失败：需检测到单张人脸", is_error=True)
            return

        # 获取检测数据
        if not (self.camera.current_prob > 0.99 and
                self.camera.liveness_status and
                self.camera.current_face_feature is not None):
            error_msg = "注册失败："
            conditions = []
            if self.camera.current_prob <= 0.99:
                conditions.append(f"置信度({self.camera.current_prob:.2f}<0.99)")
            if not self.camera.liveness_status:
                conditions.append("活体检测未通过")
            if not self.camera.current_face_feature:
                conditions.append("未获取人脸特征")
            self.status.show_message(error_msg + " | ".join(conditions), is_error=True)
            return

        try:
            # 转换特征值为可序列化格式
            embedding = self.camera.current_face_feature[0].tolist()

            # 存储用户信息
            self.registered_users[sid] = {
                "name": name,
                "password": pwd,
                "embedding": embedding,
                "register_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # 控制台输出
            print("\n当前注册用户列表：")
            for uid, data in self.registered_users.items():
                print(f"学号: {uid}")
                print(f"姓名: {data['name']}")
                print(f"注册时间: {data['register_time']}")
                print(f"特征维度: {len(data['embedding'])}")
                print(f"前5个特征值: {data['embedding'][:5]}\n")

            # 显示注册成功信息
            self.status.show_message(f"{name}({sid}) 注册成功！")

            # 自动填充登录表单
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
