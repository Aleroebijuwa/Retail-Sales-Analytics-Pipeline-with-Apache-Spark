"""Packaging configuration for the retail-sales-pipeline library."""

from setuptools import setup, find_packages

setup(
    name="retail-sales-pipeline",
    version="0.1.0",
    description="A reusable Apache Spark analytics pipeline for retail sales data.",
    author="Alero Ebijuwa",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyspark>=3.3.0,<3.6.0",
        "pandas>=1.5.0",
        "openpyxl>=3.0.0",
        "scikit-learn>=1.1.0",
        "pyarrow>=10.0.0",
    ],
    extras_require={
        "ml": ["shap>=0.44.0", "matplotlib>=3.5.0"],
        "test": ["pytest>=7.0.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
