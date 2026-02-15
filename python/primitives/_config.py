"""Rust/Python backend toggle for all primitives."""

import os

USE_RUST = os.environ.get("PRIMITIVES_USE_RUST", "1") != "0"
