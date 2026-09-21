"""Планировщик Cosmo B: блочная сеть GDS, GDE3 и неизменяемая история."""
from pathlib import Path as _Path

# Скомпилированное расширение хранится в общей сборке, исходники — здесь.
__path__.insert(0, str(_Path(__file__).resolve().parents[2] / "build" / "hackathon" / "solver"))

__version__ = "1.1.0"
