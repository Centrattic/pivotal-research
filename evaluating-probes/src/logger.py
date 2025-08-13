# src/logger.py
import sys
from pathlib import Path
from typing import Any

class Logger:
    """A simple logger that writes to a file and to the console."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        # Open the file and keep the handle
        self.log_file = open(log_path, "w", encoding='utf-8')

    def log(self, message: Any):
        """Writes a message to the console and the log file."""
        print(message, file=self.terminal, flush=True)
        print(message, file=self.log_file, flush=True)

    def close(self):
        """Closes the log file handle."""
        self.log_file.close()