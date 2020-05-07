# -*- coding: utf-8 -*-
"""setup.py"""

from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='pycalibration',
      version='0.1.0',
      description='Evaluating and testing model calibration in Python',
      long_description=readme(),
      long_description_content_type='text/markdown',
      author='David Widmann',
      author_email='david.widmann@it.uu.se',
      url='https://github.com/devmotion/pycalibration',
      classifiers=[
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Topic :: Scientific/Engineering :: Mathematics",
      ],
      license='MIT',
      packages=['pycalibration', 'pycalibration.tests'],
      data_files=[],
      install_requires=['julia>=0.2', 'POT>=0.7'],
      include_package_data=True,
      tests_require=['tox'],)
