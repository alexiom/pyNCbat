# pyNCbat

This is a package to implement the model used for NAME OF REPORT HERE by Hannah Rubenstein, Alex Marsh, Drew Van Kuiken, Jonathan Williams, and Andy Yates.

- [Website](https://alexmarsh.io/research/pyNCbat/)
- [Documentation](https://alexmarsh.io/research/pyNCbat/documentation/)

### Features

This package provides:
  - All python code for solving the model
  - Raw data?

## In This README 
- [Setup](#setup)
  - [Requirements and Dependencies](#requirements-and-dependencies)
  - [Installation](#installation)
  - [Using The Package](#using-the-package)
  - [Customization](#customization)
- [Contributing](#contributing)


## Setup

### Requirements and Dependencies
This library requires at least ```python 3.7``` and uses packages from this version's Python Standard Library. The other dependencies can be viewed [here](https://github.com/alexiom/pyNCbat/requirements/dependencies.txt).

### Installation

To install ```pyNCbat``` for python, make sure you are using a recent version of ```pip``` (```pip install --upgrade pip```) and then use ```pip install osqp```.

To install ```pyNCbat``` from source, clone [the repository](https://github.com/alexiom/pyNCbat) (```git clone https://github.com/alexiom/pyNCbat```) and run ```pip install .``` from inside the cloned folder.

### Using The Package

Examples of how to use this package can be found in the [examples](https://github.com/alexiom/pyNCbat/tree/main/examples/) directory. In this directory, the code to replicate the results in TITLE can be found in the [replicate-report-results.py](https://github.com/alexiom/pyNCbat/tree/main/examples/replicate-report-results.py) file.

### Customization

To include additional electricity generation and storage technologies customized by the user, add the necessary technology parameters to the [custom-techs.py](https://github.com/alexiom/pyNCbat/tree/main/pyNCbat/parameters/custom-techs.py) file in the [parameters directory](https://github.com/alexiom/pyNCbat/tree/main/pyNCbat/parameters/).

## Contributing


