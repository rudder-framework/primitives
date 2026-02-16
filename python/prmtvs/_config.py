"""Rust/Python backend toggle for all prmtvs."""

import os

USE_RUST = os.environ.get("PRMTVS_USE_RUST", "1") != "0"
