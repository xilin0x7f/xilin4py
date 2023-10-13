# Author: 赩林, xilin0x7f@163.com
from setuptools import setup, find_packages

setup(
    name='xilin4py',
    version='0.11',
    description='A sample Python package',
    author='xilin0x7f',
    author_email='xilin0x7f@163.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
    ],
)
