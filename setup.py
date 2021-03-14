import os
from distutils.core import setup
import sys

setup(
    name = "kilean_utils",
    version = "0.0.1",
    author = "Kilean Hwang",
    author_email = "kilean@lbl.gov",
    description = ("utilities"),
    license = "Lawrence Berkeley National Laboratory",
    keywords = "IMPACT",
    url = "",
    packages=['kilean_utils'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Topic :: Utilities",
        "License :: Free for non-commercial use",
    ],
    zip_safe=False
)
