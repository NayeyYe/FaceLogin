# src-utils/logger.py
import sys
import os
import atexit
from datetime import datetime
from types import TracebackType
from typing import Optional, Type
import logging
from config import logscfg


def setup_logger(cfg):
    """配置日志系统"""
    log_dir = logscfg.logs_dir
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # 文件处理器
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式设置
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class SessionLogger:
    def __init__(self):
        self.file_handle = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.session_start = None

    def configure(self, log_dir=logscfg.logs_dir, log_file="log.txt") -> None:
        """配置日志系统"""
        os.makedirs(log_dir, exist_ok=True)
        self.file_handle = open(os.path.join(log_dir, log_file), "a", encoding="utf-8")
        self.session_start = datetime.now()

        # 注册清理回调
        atexit.register(self.close_session)
        sys.excepthook = self._exception_hook

    def _exception_hook(
            self,
            exc_type: Type[BaseException],
            exc_value: BaseException,
            exc_traceback: Optional[TracebackType]
    ) -> None:
        """自定义异常处理钩子"""
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        self.close_session()

    def close_session(self) -> None:
        """关闭日志会话"""
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.write("\n" + "-" * 40 + "\n")
            self.file_handle.close()

        # 恢复原始输出流
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


class LogRedirector:
    def __init__(self, original_stream, logger: SessionLogger):
        self.original_stream = original_stream
        self.logger = logger

    def write(self, message: str) -> None:
        """双路写入"""
        if message.strip() and self.logger.file_handle:
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            self.logger.file_handle.write(f"{timestamp}{message}")
        self.original_stream.write(message)

    def flush(self) -> None:
        self.original_stream.flush()


def enable_logging() -> SessionLogger:
    """启用日志系统"""
    logger = SessionLogger()
    logger.configure()

    # 重定向标准输出
    sys.stdout = LogRedirector(sys.stdout, logger)
    sys.stderr = LogRedirector(sys.stderr, logger)

    return logger


# test.py 使用示例
if __name__ == "__main__":
    logger = enable_logging()

    print("正常操作日志")
    print(1/0)  # 制造错误

