from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gen-dedup",
    version='0.0.1',
    author="Sean Lee",
    author_email="xmlee97@gmail.com",
    description="Generative deduplication",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/4AI/generative_deduplication",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "tqdm",
        "datasets",
        "transformers",
    ],
)
