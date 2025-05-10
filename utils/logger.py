# logger.py
import os
import sys
import logging
import atexit
from datetime import datetime
import inspect
from config import logscfg


class EnhancedLogger:
    def __init__(self):
        self.logger = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.caller_file = None
        self.start_time = None
        self._setup_complete = False

    def _get_caller_filename(self):
        """获取调用logger的源文件名"""
        stack = inspect.stack()
        for frame in stack:
            filename = frame.filename
            if filename != __file__:
                return os.path.basename(filename)
        return "unknown_source"

    def _setup_logfile(self):
        """配置日志文件路径"""
        log_dir = os.path.dirname(logscfg.logs)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        return logscfg.logs

    def _create_formatters(self):
        """创建自定义日志格式"""
        return logging.Formatter(
            f"[%(asctime)s] [%(levelname)-8s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    class StreamRedirector:
        """重定向标准输出/错误到日志系统"""

        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.buffer = []

        def write(self, message):
            if message.strip():
                self.buffer.append(message.strip())

        def flush(self):
            if self.buffer:
                self.logger.log(self.level, " ".join(self.buffer))
                self.buffer = []

    def _init_logger(self):
        """初始化日志系统核心"""
        self.logger = logging.getLogger("GlobalLogger")
        self.logger.setLevel(logging.DEBUG)

        # 防止重复添加处理器
        if self.logger.handlers:
            return

        # 文件处理器配置
        file_handler = logging.FileHandler(self._setup_logfile())
        file_handler.setFormatter(self._create_formatters())

        # 控制台处理器配置
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self._create_formatters())

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _register_hooks(self):
        """注册系统钩子"""
        # 标准输出重定向
        sys.stdout = self.StreamRedirector(self.logger, logging.INFO)
        sys.stderr = self.StreamRedirector(self.logger, logging.ERROR)

        # 异常捕获钩子
        def exception_handler(exc_type, exc_value, exc_traceback):
            self.logger.error(
                "未捕获的异常",
                exc_info=(exc_type, exc_value, exc_traceback)
            )

        sys.excepthook = exception_handler

        # 退出处理函数
        def exit_handler():
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.logger.info(f"运行结束时间: {end_time}")
            self.logger.info("-" * 60 + "\n")
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

        atexit.register(exit_handler)

    def initialize(self):
        """初始化日志系统"""
        if self._setup_complete:
            return

        self.caller_file = self._get_caller_filename()
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._init_logger()
        self._register_hooks()

        # 写入初始分隔符和启动信息
        self.logger.info("\n" + "-" * 60)
        self.logger.info(f"正在运行文件: {self.caller_file}")
        self.logger.info(f"运行开始时间: {self.start_time}")
        self.logger.info("-" * 60)

        self._setup_complete = True


# 单例模式实例化
logger = EnhancedLogger().initialize()
