"""测试套件。用标准库 unittest（pytest 亦可直接运行：pytest tests/）。"""
import os
import sys

# 让 tests 不依赖安装：把 src 加进路径
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
