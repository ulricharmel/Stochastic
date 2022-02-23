import os
from setuptools import setup, find_packages
from stochastic import __version__

build_root = os.path.dirname(__file__)

def requirements():
    """Get package requirements"""
    with open(os.path.join(build_root, 'requirements.txt')) as f:
        return [pname.strip() for pname in f.readlines()]

with open("README.rst") as tmp:
    readme = tmp.read()

console_scripts = ['stochastic=stochastic.main:main', 'stochastic-opt=stochastic.opt_hyper_params:main', 
                    'stochastic-prep=stochastic.preprocess.wsclean_model:main']

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
    install_requires=requirements(),
    tests_require=["requests", "pytest", "numpy"],
    packages=find_packages(include=['stochastic','stochastic.*']),
    entry_points={
        'console_scripts': console_scripts
    },
    keywords='stochastic',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3.9',
        ],
)
