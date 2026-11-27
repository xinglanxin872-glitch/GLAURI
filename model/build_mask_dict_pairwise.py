#!/usr/bin/env python3
"""Compatibility entry for the corrected CA pairwise-mask builder."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.build_masks import main
if __name__=='__main__':main()
