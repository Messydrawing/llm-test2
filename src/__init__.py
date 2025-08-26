"""Proxy package exposing ``test7/src`` modules for unit tests."""

import pathlib

__path__ = [str(pathlib.Path(__file__).resolve().parent.parent / "test7" / "src")]
