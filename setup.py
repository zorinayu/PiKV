from setuptools import setup, find_packages

setup(
    name="pikv",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tqdm>=4.65.0",
    ],
) 