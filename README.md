# 人脸识别系统 MTCNN+FaceNet

一个基于PyQt5的实时人脸检测与识别系统，支持摄像头活体验证、用户注册登录等功能。

## 主要功能

- **实时检测**：摄像头人脸检测框与5个关键点
- **活体检测**：通过头部姿态分析判断活体
- **用户管理**：
  - 注册新用户（需人脸验证）
  - 登录验证（人脸+密码双重验证）
- **多模式切换**：支持OpenCV/MTCNN两种检测方式
- **可视化界面**：
  - 实时FPS显示
  - 人脸计数
  - 置信度/姿态角度显示

##  环境安装

原本是有requirements.txt用于版本控制，但是版本过高了有些设备无法下载，所以下列的库版本只用于参考

**python == 3.9**

| 库            | 版本         |
| ------------- | ------------ |
| torch         | 2.7.0+cu126  |
| torchvision   | 0.22.0+cu126 |
| torchsummary  | 1.5.1        |
| scipy         | 1.13.1       |
| numpy         | 1.26.3       |
| opencv_python | 4.11.0.86    |
| pillow        | 10.2.0       |
| dlib          | 19.23.0      |
| matplotlib    | 3.9.4        |
| tqdm          | 4.67.1       |
| pyqt5         | 5.15.11      |
| pymysql       | 1.1.1        |
| cryptography  | 41.0.2       |
| bcrypt        | 4.1.1        |

* 关于dlib的安装

先要下载相关包**boost， cmake，opencv_python**

```bash
pip install boost -i https://mirrors.aliyun.com/pypi/simple/

pip install cmake-i https://mirrors.aliyun.com/pypi/simple/

pip install opencv-python -i https://mirrors.aliyun.com/pypi/simple/
```

然后再清华源https://pypi.tuna.tsinghua.edu.cn/simple/dlib/中下载dlib对应的python版本

* 如：python == 3.9 ，再whl文件上面就是cp39，以此类推
* 然后在你whl文件的目录下输入命令

```bash
pip install dlib-19.23.0-cp39-cp39-win_amd64.whl
```

* 下载shape_predictor_68_face_landmarks.dat到你的根目录下
  * 这是一个人脸识别的检测模型，不下载的话就无法使用dlib人脸68点检测
  * https://dlib.net/imaging.html#shape_predictor官网下载即可

## 项目结构

```text
├── FaceLogin/ (主项目目录)
│   ├── dataset/
│   │   ├── 12/                  # generate_PNet_data.py生成
│   │   ├── 24/                  # generate_RNet_data.py生成
│   │   ├── 48/                  # generate_ONet_data.py生成
│   │   ├── lfw_5590/            # 自行下载
│   │   ├── net_7876/            # 自行下载
│   │   ├── WIDER_train/         # 自行下载
│   │   ├── testImageList.txt    # 下载lfw_5590/和net_7876/自带
│   │   ├── trainImageList.txt   # 下载lfw_5590/和net_7876/自带
│   │   └── wider_face_train.txt # 源代码中有，是wider_split的增强版
│   │   
│   ├── db/ (数据库相关文件)
│   │   ├──csv_syncer.py (0.0B)         # CSV与数据库同步工具
│   │   ├── db_init.py (1.6KB)       # 数据库初始化脚本
│   │   ├── database_manager.py (3.7KB)  # 数据库连接和CRUD操作
│   │   ├── info.csv (0.0B)          # 用户信息存储CSV
│   │   └── test.sql (0.0B)          # SQL测试脚本
│   │
│   ├── detect/ (人脸检测相关)
│   │   ├── infer_camera.py (8.7KB)  # 摄像头实时推理
│   │   ├── infer_path.py (8.7KB)    # 图片路径批量推理
│   │   ├── liveness.py (4.6KB)      # 活体检测模块
│   │   └── mtcnn.py (5.8KB)         # MTCNN人脸检测实现
│   │   └── (1目录)
│   │
│   ├── gui/ (图形界面相关)
│   │   ├── GUI.py (9.3KB)           # 主界面窗口
│   │   ├── camera.py (7.3KB)        # 摄像头画面处理
│   │   ├── control_button.py (3.2KB) # 控制按钮组件
│   │   ├── login_form.py (4.2KB)    # 登录表单组件
│   │   └── status.py (5.0KB)        # 状态显示组件
│   │   └── (1目录)
│   │
│   ├── img/ (图片资源)
│   │   └── test.jpg (141.6KB)       # 测试用图片
│   │
│   ├── models/ (预训练模型)
│   │   ├── ONet.pth (1.5MB)         # MTCNN第三阶段模型
│   │   ├── PNet.pth (44.5KB)        # MTCNN第一阶段模型
│   │   └── RNet.pth (414.2KB)       # MTCNN第二阶段模型
│   │
│   ├── train_Net/ (模型训练相关)
│   │   ├── generate_ONet_data.py (8.1KB)  # ONet训练数据生成
│   │   ├── generate_PNet_data.py (8.6KB)  # PNet训练数据生成
│   │   ├── generate_RNet_data.py (5.4KB)  # RNet训练数据生成
│   │   ├── model.py (8.0KB)         # 模型架构定义
│   │   ├── train_ONet.py (3.2KB)    # ONet训练脚本
│   │   ├── train_PNet.py (3.2KB)    # PNet训练脚本
│   │   └── train_RNet.py (3.2KB)    # RNet训练脚本
│   │
│   └── utils/ (工具函数)
│       ├── data.py (2.8KB)          # 数据加载/预处理工具
│       ├── data_format_converter.py (2.6KB)  # 数据格式转换
│       ├── encryption.py (2.5KB)    # 加密解密工具
│       └── utils.py (28.4KB)        # 通用工具函数
│   
│
├── config.py (1.8KB)                # 全局配置文件
├── Test.py (3.7KB)                  # 单元测试脚本
├── README.md (5.9KB)                # 项目说明文档
└── shape_predictor_68_face_landmarks.dat (95.1MB)  # 人脸关键点检测模型

```

## 使用

1. **准备数据集** (需要提前下载)
   ```bash
   # 创建数据集目录
   mkdir dataset && cd dataset
    
   # 下载并解压WIDER Face数据集
   # 下载WIDER_train就可以了
   http://shuoyang1213.me/WIDERFACE/
   # 下载lfw_5590数据集
   # 下载net_7876数据集
   http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip
   # lfw_5590和net_7876文件夹都是存放人脸图片的
   # testImageList.txt和trainImageList.txt都是标注信息文本文件，标注信息为图片文件、人脸box的坐标位置、人脸5个关键点的坐标位置
   ```

   最后得到的dataset中应该有

   * WIDER_train
   * lfw_5590
   * net_7876
   * testImageList.txt
   * trainImageList.txt
   * wider_face_train.txt（这个是本人提供的，在源代码中）

2. **训练模型** (可选)

   ```bash
   # 依次启动下面的代码
   python generate_Pnet_data.py
   python train_Pnet.py
   python generate_Rnet_data.py
   python train_Rnet.py
   python generate_Onet_data.py
   python train_Onet.py
   ```

   运行完成后在dataset文件夹中有**12，24，48**三个文件夹，里面是图片的标签文件（切分之后的文件在处理完标签之后已经被删除，也可以不删除，只要把generate代码中的最后一行注释掉就可以了）

3. **运行主程序**

   * 初始化mysql

   ```bash
   python db_init.py
   ```

   * 运行GUI.py即可

   ```bash
   python GUI.py
   ```

   

## 界面操作指南

1. **摄像头控制**：
   -  启动/关闭摄像头
   - 开启/关闭人脸检测

2. **用户管理**：
   -  注册：输入姓名学号 -> 正对摄像头 -> 点击注册
   -  登录：输入学号密码 -> 正对摄像头 -> 点击登录

3. **显示信息**：
   - 实时人脸置信度
   - 头部姿态角度（翻滚/俯仰/偏航）
   - 活体检测状态
