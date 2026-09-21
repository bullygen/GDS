"""Проверяет, что исходный и двоичный дистрибутивы не содержат локальные материалы."""
from pathlib import Path
import tarfile
import zipfile


def verify(names):
    """Отвергает рабочие задания, статьи, кэши, окружения и прежние модули."""
    forbidden = ("prompt.md", "articles/", "outputs/", "__pycache__/", ".pytest_cache/",
                 ".venv/", ".git/", "MODEL_", "UTILS_")
    for name in names:
        assert not any(part in name for part in forbidden), name
        assert not name.endswith((".pyc", ".pdf")), name


def main():
    """Проверяет оба вида пакета после выполнения python -m build."""
    source_archives = list(Path("dist").glob("*.tar.gz"))
    wheels = list(Path("dist").glob("*.whl"))
    assert source_archives and wheels, "Сначала выполните python -m build."
    for filename in source_archives:
        with tarfile.open(filename) as archive:
            names = archive.getnames()
            verify(names)
            assert any(name.endswith("src/scheduling.cpp") for name in names)
            assert any(name.endswith("tests/test_core.cpp") for name in names)
            print(filename.name, "состав исходного архива проверен")
    for filename in wheels:
        with zipfile.ZipFile(filename) as archive:
            names = archive.namelist()
            verify(names)
            assert any("gds/_core" in name for name in names)
            print(filename.name, "состав двоичного пакета проверен")


if __name__ == "__main__":
    main()
