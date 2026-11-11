"""stratlab.catalog — every module in this package is imported, and registers its strategies on import.
To add strategies, drop a new .py file here (see ../README.md); no other file needs editing."""
import importlib
import pkgutil

for _m in sorted(m.name for m in pkgutil.iter_modules(__path__)):
    if not _m.startswith("_"):
        importlib.import_module(f"{__name__}.{_m}")
