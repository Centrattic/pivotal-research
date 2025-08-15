# src/logger.py
import sys
import threading
from pathlib import Path
from typing import Any


class Logger:
    """A simple logger that writes to a file and to the console."""

    def __init__(
        self,
        log_path: Path,
    ):
        self.terminal = sys.stdout
        # Shared lock across all Logger instances to avoid interleaved writes
        if not hasattr(Logger, "_lock"):
            Logger._lock = threading.Lock()
        # Open the file in append mode to avoid truncation when multiple loggers are created
        self.log_file = open(log_path, "a", encoding='utf-8')

    def log(
        self,
        message: Any,
    ):
        """Writes a message to the console and the log file."""
        print(message, file=self.terminal, flush=True)
        with Logger._lock:
            print(message, file=self.log_file, flush=True)

    def close(
        self,
    ):
        """Closes the log file handle."""
        self.log_file.close()
