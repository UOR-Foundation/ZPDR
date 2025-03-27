"""
Setup script for the Zero-Point Data Resolution (ZPDR) package.
"""

from setuptools import setup, find_packages

setup(
    name="zpdr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    author="ZPDR Team",
    author_email="info@zpdr.org",
    description="Zero-Point Data Resolution - A mathematical framework for universal data encoding and resolution",
    long_description=open("Pure-ZPDR.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zpdr/zpdr",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
)