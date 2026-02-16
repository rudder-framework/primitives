"""Rust/Python backend toggle for all pmtvs."""

import os

USE_RUST = os.environ.get("PMTVS_USE_RUST", "1") != "0"
