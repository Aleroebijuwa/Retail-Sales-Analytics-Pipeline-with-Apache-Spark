"""Utility helpers for the retail pipeline package."""

import os

import pyspark


def configure_spark_home():
    """Point ``SPARK_HOME`` at the installed PySpark package.

    Some machines have a stale ``SPARK_HOME`` environment variable that points
    at a non-existent Spark install, which stops Spark from launching. Setting
    it to the pip-installed PySpark location makes session creation reliable
    (especially on Windows).

    Returns
    -------
    str
        The resolved ``SPARK_HOME`` path.
    """
    home = os.path.dirname(pyspark.__file__)
    if os.environ.get("SPARK_HOME") != home:
        os.environ["SPARK_HOME"] = home
    return home
