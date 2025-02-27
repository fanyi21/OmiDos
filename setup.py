from setuptools import setup
from setuptools import find_packages

VERSION = '0.1.0'

setup(
    name='OmiDos',
    version=VERSION,
    description='we introduce OmiDos, a novel dynamic orthogonal deep generative model, specifically designed to disentangle shared and unique molecular signatures across multi-omics layers in single-cell multi-omics data.',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy',
    ],
    python_requires='>=3.6',
    author='Yi Fan',
    )