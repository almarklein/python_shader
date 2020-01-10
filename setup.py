import re

from setuptools import find_packages, setup


with open("python_shader/__init__.py") as fh:
    VERSION = re.search(r"__version__ = \"(.*?)\"", fh.read()).group(1)

description = "Write modern GPU shaders in Python!"
url = "https://github.com/almarklein/python-shader"

setup(
    name="python-shader",
    version=VERSION,
    description=description,
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.6.0",
    license=open("LICENSE").read(),
    # long_description=open("README.md").read(),
    long_description=description + "\n\n" + url,
    long_description_content_type="text/markdown",
    author="Almar Klein",
    author_email="almar.klein@gmail.com",
    url=url,
    zip_safe=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
