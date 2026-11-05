"""Entry point of the project for the Flask application (Slide 9 & 20)."""

import sys
from pathlib import Path

# Ensure root project directory is in python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.conf.config import get_default
from src.utils.logger import get_logger
from src.template_app.app import create_app

logger = get_logger("run_app")

if __name__ == "__main__":
    config = get_default()
    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 5000)
    debug = server_cfg.get("debug", False)

    logger.info(f"Starting Flask API Server on {host}:{port}...")
    app = create_app()
    app.run(host=host, port=port, debug=debug)
