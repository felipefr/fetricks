import setuptools
import os


setuptools.setup(
    name='fetricks',
    author="Felipe Rocha",
    author_email="felipe.figueredo-rocha@ec-nantes.fr",
    version="0.1.0",
    packages=setuptools.find_packages(include=['fetricks', 'fetricks.*']),
    url='https://github.com/felipefr/fetricks',
    license='MIT License',
    description='Finite elements tricks using FEniCS'
)
