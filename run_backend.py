import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
venv_site_packages = ROOT / ".venv" / "Lib" / "site-packages"
if venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

import uvicorn


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=False)
