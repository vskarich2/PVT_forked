import time
import torch
import torch.nn as nn

# Assume PVT, DSVA_Block, SBoxBlock, RelativeAttention, etc. are defined elsewhere
# from your_codebase import PVT, DSVA_Block, SBoxBlock, RelativeAttention


import threading
import time
import yaml
import os
from typing import Any, Dict, Optional


class ConfigWatcher:
    """
    Watches a YAML config file on disk and reloads it every `reload_interval` seconds.

    Usage:
        # In your main program’s entrypoint:
        watcher = ConfigWatcher("path/to/config.yaml", reload_interval=10.0)
        watcher.start()           # spawn background thread to reload every 10 sec

        # Later, anywhere in your code:
        config = watcher.get()    # returns the most recently loaded dictionary
        if config.get("disable_dynamic", False):
            # your logic here

        # To shut down at program exit:
        watcher.stop()
    """

    def __init__(self, filepath: str, reload_interval: float = 10.0):
        """
        Args:
            filepath: path to the YAML config file to watch.
            reload_interval: how often (in seconds) to re‐load the file.
        """
        self.filepath = filepath
        self.reload_interval = reload_interval
        self._lock = threading.Lock()
        self._config: Dict[str, Any] = {}
        self._last_mtime: Optional[float] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Attempt an initial load
        self._load_config()

    def _load_config(self) -> None:
        """
        Loads the YAML file if it has been modified since last load.
        If the file doesn't exist or is invalid, logs an error but does not raise.
        """
        try:
            mtime = os.path.getmtime(self.filepath)
        except FileNotFoundError:
            # File not yet created or temporarily missing
            return

        # Only reload if the file on disk has a newer modification time
        if self._last_mtime is not None and mtime <= self._last_mtime:
            return

        try:
            with open(self.filepath, "r") as f:
                new_conf = yaml.safe_load(f) or {}
            if not isinstance(new_conf, dict):
                raise ValueError(f"Config file {self.filepath} did not parse as a dict")
        except Exception as e:
            print(f"[ConfigWatcher] Error loading config file '{self.filepath}': {e!r}")
            return

        # Atomically swap in the new config
        with self._lock:
            self._config = new_conf
            self._last_mtime = mtime
            # Optionally, you can print/log whenever we successfully reload:
            print(f"[ConfigWatcher] Reloaded config from '{self.filepath}' at {time.ctime(mtime)}")

    def _watch_loop(self) -> None:
        """
        Background thread target: every `reload_interval` seconds, attempt to reload.
        """
        while not self._stop_event.is_set():
            self._load_config()
            # Sleep, but wake up early if stop() is called
            self._stop_event.wait(self.reload_interval)

    def start(self, daemon: bool = True) -> None:
        """
        Start the background thread that watches the file.
        This is idempotent: calling start() multiple times does nothing after the first.
        """
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=daemon)
        self._thread.start()
        print(f"[ConfigWatcher] Started watching '{self.filepath}' every {self.reload_interval}s")

    def stop(self) -> None:
        """
        Stop the background thread. Once stopped, the thread will exit cleanly.
        """
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.reload_interval + 1.0)
        self._thread = None
        print(f"[ConfigWatcher] Stopped watching '{self.filepath}'")

    def get(self) -> Dict[str, Any]:
        """
        Thread‐safe way to retrieve the current config dictionary.
        Returns a shallow copy so callers cannot mutate the internal state.
        """
        with self._lock:
            return dict(self._config)  # shallow copy

    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Shortcut for retrieving one value from the config.
        """
        with self._lock:
            return self._config.get(key, default)


# If someone runs this module directly, demonstrate how it works.
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python config_watcher.py path/to/config.yaml")
        sys.exit(1)

    path_to_yaml = sys.argv[1]
    watcher = ConfigWatcher(path_to_yaml, reload_interval=10.0)
    watcher.start()

    try:
        while True:
            cfg = watcher.get()
            print(f"[Main] Current config: {cfg}")
            time.sleep(10.0)
    except KeyboardInterrupt:
        print("\n[Main] Caught Ctrl+C, stopping watcher.")
        watcher.stop()


# For demo, let's define dummy attention modules that check the flags
class DummyDynamicAttention(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.config = config
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        # Suppose x is [B, M, D]
        if self.config.get_value("disable_dynamic", False):
            # Bypass dynamic attention
            return x
        return torch.relu(self.lin(x))


class DummyWindowAttention(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.config = config
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        if self.config.get_value("disable_window", False):
            return x
        return torch.relu(self.lin(x))


class DummyGlobalAttention(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.config = config
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        if self.config.get_value("disable_global", False):
            return x
        return torch.relu(self.lin(x))

class SimpleModel(nn.Module):
    def __init__(self, config_watcher: ConfigWatcher, input_dim=64):
        super().__init__()
        self.conf = config_watcher
        self.dynamic_attn = DummyDynamicAttention(config_watcher, input_dim)
        self.window_attn  = DummyWindowAttention(config_watcher, input_dim)
        self.global_attn  = DummyGlobalAttention(config_watcher, input_dim)
        self.classifier  = nn.Linear(input_dim, 10)

    def forward(self, x):
        # x: [B, M, D]
        x = self.dynamic_attn(x)
        x = self.window_attn(x)
        x = self.global_attn(x)

        # Suppose we simply do a global‐pool then classify
        x = x.mean(dim=1)  # [B, D]
        return self.classifier(x)

if __name__ == "__main__":
    # 1) Instantiate and start the watcher
    watcher = ConfigWatcher("example_config.yaml", reload_interval=10.0)
    watcher.start()

    # 2) Build a dummy model that reads flags from the watcher
    model = SimpleModel(watcher, input_dim=64)

    # 3) Dummy data
    batch_size = watcher.get_value("batch_size", 32)
    dummy_input = torch.randn(batch_size, 100, 64)  # [B, M=100 voxels, D=64]

    # 4) Training / inference loop
    try:
        for step in range(1, 101):
            # Each iteration, re‐read learning rate (could have changed on disk)
            lr = watcher.get_value("learning_rate", 0.001)
            # (re)create optimizer with current learning rate, or adjust lr of existing optim:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Forward pass
            logits = model(dummy_input)
            loss = logits.pow(2).mean()  # some dummy loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print out which attentions are currently enabled/disabled
            cfg = watcher.get()
            print(f"[Step {step}] loss={loss.item():.4f} | "
                  f"disable_dynamic={cfg.get('disable_dynamic', False)} | "
                  f"disable_window={cfg.get('disable_window', False)} | "
                  f"disable_global={cfg.get('disable_global', False)} | "
                  f"lr={lr:.5f}")

            time.sleep(1.0)  # simulate batch‐by‐batch delay

    except KeyboardInterrupt:
        print("[Main] Interrupted by user. Shutting down…")
    finally:
        watcher.stop()
