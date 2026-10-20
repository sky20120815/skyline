"""让 pytest 在未安装包的情况下也能 import src/（CI 里 `pip install -e .` 后此文件无副作用）。"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
