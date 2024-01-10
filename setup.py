import setuptools
import os


setuptools.setup(
    name='fetricksx',
    author="Felipe Rocha",
    author_email="felipe.figueredo-rocha@u-pec.fr",
    version="0.1.0",
    packages=setuptools.find_packages(include=['fetricksx', 'fetricksx.*']),
    url='https://github.com/felipefr/fetricks',
    license='MIT License',
    description='Finite elements tricks using FEniCSx'
)
