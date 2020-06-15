from setuptools import find_packages, setup


def get_version_and_docstring():
    ns = {"__doc__": "", "__version__": ""}
    docStatus = 0  # Not started, in progress, done
    for line in open("pyshader/__init__.py").readlines():
        if line.startswith("__version__"):
            exec(line.strip(), ns, ns)
        elif line.startswith('"""'):
            if docStatus == 0:
                docStatus = 1
                line = line.lstrip('"')
            elif docStatus == 1:
                docStatus = 2
        if docStatus == 1:
            ns["__doc__"] += line.rstrip() + "\n"
    return ns["__version__"], ns["__doc__"]


version, doc = get_version_and_docstring()

setup(
    name="pyshader",
    version=version,
    url="https://github.com/pygfx/pyshader",
    description="Write modern GPU shaders in Python!",
    long_description=doc,
    long_description_content_type="text/markdown",
    author="Almar Klein",
    author_email="almar.klein@gmail.com",
    packages=find_packages(
        exclude=["tests", "tests.*", "examples_py", "examples_py.*"]
    ),
    python_requires=">=3.6.0",
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
