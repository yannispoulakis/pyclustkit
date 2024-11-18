#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gettext import install

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    readme = fh.read()

with open("requirements.txt") as req_file:
    install_requires = req_file.read().splitlines()
print(install_requires)
setup(
    # Meta-data and description
    name='pyclustkit',
    version='0.1.0',
    description='A Python library for clustering operations. Evaluation and meta-feature generation.',
    author='Yannis Poulakis',
    author_email='giannispoy@gmail.com',
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/yannispoulakis/pyclustkit",

    # Packages and requirements
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.6',
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    # Console for deepwalk
    entry_points={'console_scripts': ['deepwalk = deepwalk_custom.__main__:main']},
    package_dir={'deepwalk_custom': 'deepwalk_custom'},

    zip_safe=False,
    keywords=["Clustering", "Meta-Learning", "Meta-Features", "Evaluation"],
    test_suite='tests'
)
