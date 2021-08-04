from setuptools import setup, find_packages
from stochastic import __version__

with open("README.rst") as tmp:
    readme = tmp.read()

setup(
    author='Ulrich A. Mbou Sob',
    author_email='mulricharmel@gmail.com',
    name='stochastic',
    version=__version__,
    description='Fitting source models to visibilities using stochastic optimisation',
    long_description=readme,
    long_description_content_type="text/x-rst",
    url='https://github.com/ulricharmel/Stochastic',
    license='GNU GPL v2',
    packages=find_packages(include=['stochastic','stochastic.*']),
    entry_points={
        'console_scripts': ['stochastic=stochastic.main:main']
    },
    keywords='stochastic',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3.9',
        ],
)
