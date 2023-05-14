#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
FILE: setup.py
PROJECT: simulations
ORIGINAL AUTHOR: ryancardenas
DATE CREATED: 12 May 2023
"""

from setuptools import setup, find_packages


setup(
    name="simulations",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy >= 1.22.3"],
    tests_require=["pytest"],
)
