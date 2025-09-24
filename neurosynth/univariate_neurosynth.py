#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin runner for univariate Neurosynth analysis.

All functions live in `neurosynth/modules`. This script only orchestrates the run.
"""

from neurosynth.modules.univariate_main import main


if __name__ == "__main__":
    main()

