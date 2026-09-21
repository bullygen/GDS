"""Сборка физического охранного оператора и сети из исходников проекта."""
from pathlib import Path
import subprocess
import sys


def build():
    import pybind11
    root = Path(__file__).resolve().parent
    subprocess.run([
        "cmake", "-S", str(root), "-B", str(root / "build"),
        "-DCMAKE_BUILD_TYPE=Release", f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
    ], check=True)
    subprocess.run(["cmake", "--build", str(root / "build"), "--parallel", "2"], check=True)


if __name__ == "__main__":
    build()
