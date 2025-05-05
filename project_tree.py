import os
from pathlib import Path


class FileTreeGenerator:
    """带过滤功能的项目文件树生成器"""

    def __init__(self,
                 root_path: str,
                 ignore_dirs: set = None,
                 ignore_hidden: bool = True,
                 max_depth: int = 3,
                 show_file_counts: bool = True):
        # 初始化配置参数
        self.root_path = Path(root_path).resolve()
        self.ignore_dirs = ignore_dirs or {'__pycache__', '.git', 'venv', '.idea', 'node_modules'}
        self.ignore_hidden = ignore_hidden
        self.max_depth = max_depth
        self.show_file_counts = show_file_counts

        # 初始化统计信息
        self.stats = {'dirs': 0, 'files': 0}

    def generate(self) -> None:
        """生成并打印文件树"""
        print(f"项目结构：{self.root_path}")
        self._process_directory(self.root_path)
        print(f"\n统计：{self.stats['dirs']}个目录，{self.stats['files']}个文件")

    def _process_directory(self, path: Path, prefix: str = '', depth: int = 0) -> None:
        """处理单个目录"""
        if depth > self.max_depth:
            return

        if not self._should_display(path):
            return

        self._print_directory(path, prefix, depth)
        self.stats['dirs'] += 1

        try:
            entries = self._get_sorted_entries(path)
            for index, entry in enumerate(entries):
                self._process_entry(entry, prefix, depth, index, len(entries))
        except PermissionError:
            print(f"{prefix}    [权限拒绝访问]")

    def _should_display(self, path: Path) -> bool:
        """判断是否应该显示该路径"""
        name = path.name
        if self.ignore_hidden and name.startswith('.'):
            return False
        if path.is_dir() and name in self.ignore_dirs:
            return False
        return True

    def _get_sorted_entries(self, path: Path) -> list:
        """获取排序后的目录条目"""
        entries = []
        try:
            entries = sorted(os.scandir(path), key=lambda x: (not x.is_dir(), x.name))
            return [e for e in entries if self._should_display(Path(e.path))]
        except PermissionError:
            return []

    def _process_entry(self, entry, prefix: str, depth: int, index: int, total: int) -> None:
        """处理目录中的单个条目"""
        is_last = index == total - 1
        connector = '    ' if is_last else '│   '
        new_prefix = f"{prefix}{connector}"

        if entry.is_dir():
            self._process_directory(Path(entry.path), new_prefix, depth + 1)
        else:
            self._print_file(entry, prefix, is_last)
            self.stats['files'] += 1

    def _print_directory(self, path: Path, prefix: str, depth: int) -> None:
        """打印目录信息"""
        is_root = (depth == 0)
        line = f"{prefix}{'└── ' if is_root else '├── '}{path.name}/"
        if self.show_file_counts:
            line += self._get_directory_stats(path)
        print(line)

    def _print_file(self, entry, prefix: str, is_last: bool) -> None:
        """打印文件信息"""
        connector = '└── ' if is_last else '├── '
        line = f"{prefix}{connector}{entry.name}"
        if self.show_file_counts:
            line += f" ({self._human_size(entry.stat().st_size)})"
        print(line)

    def _get_directory_stats(self, path: Path) -> str:
        """获取目录统计信息"""
        try:
            entries = list(os.scandir(path))
            file_count = sum(1 for e in entries if e.is_file())
            dir_count = sum(1 for e in entries if e.is_dir())
            return f" ({file_count}文件/{dir_count}目录)"
        except PermissionError:
            return " (无法统计)"

    @staticmethod
    def _human_size(size: int) -> str:
        """转换文件大小为易读格式"""
        units = ['B', 'KB', 'MB', 'GB']
        for unit in units:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


# 使用示例
if __name__ == "__main__":
    generator = FileTreeGenerator(
        root_path=".",
        ignore_dirs={'Test', 'Train', '__pycache__', '.git', 'venv', '.idea', 'node_modules', 'dataset', 'logs', 'reports', 'assets'},
        max_depth=5,
        show_file_counts=True
    )
    generator.generate()
