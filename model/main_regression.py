#!/usr/bin/env python3
"""Compatibility entry: use the unified config-only paper trainer."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.train import main
if __name__=='__main__':main()
