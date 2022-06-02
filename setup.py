from setuptools import setup

####################################
# Add long description from README #
####################################

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


#########
# Setup #
#########

setup(
    name="wakis",
    version="0.0.7",
    description="Wakis obtains Wake potential and Impedance from pre-computed fields",
    author="Elena de la Fuente Garcia",
    author_email="elena.de.la.fuente.garcia@cern.ch",
    packages=['wakis'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ImpedanCEI/WAKIS",
    project_urls={"Bug Tracker": "https://github.com/ImpedanCEI/WAKIS/issues"},
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'scikit-image >= 0.17',
        'h5py',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],

)
