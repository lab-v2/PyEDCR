<p align="center">
    <img alt="Logo" src="images/logo-transparent-png.png" width="440" height="500"/>
</p>

<!-- [![pages-build-deployment](https://github.com/krichelj/PyDiffGame/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/krichelj/PyDiffGame/actions/workflows/pages/pages-build-deployment) -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

- [What is this?](#what-is-this)
- [Example](#example)

# What is this?

[`PyEDCR`](https://github.com/lab-v2/metacognitive_error_detection_and_correction_v2/tree/master) is a Python implementation of the f-EDR (Focused Error Detection Rules) paradigm. The goal of EDR is to use a set of conditions to learn when a machine learning model makes an incorrect prediction, as first introduced in:

- https://arxiv.org/pdf/2308.14250.pdf

The package was tested for Python >= 3.9.

# Example

To demonstrate the use of the package, consider the following running example using the 'run_experiment' function from PyEDCR.py

```python
import experiment_config
from PyEDCR import run_experiment

military_vehicles_config = experiment_config.ExperimentConfig(
        data_str='military_vehicles',
        main_model_name='vit_b_16',
        secondary_model_name='vit_l_16',
        main_lr=0.0001,
        secondary_lr=0.0001,
        binary_lr=0.0001,
        original_num_epochs=10,
        secondary_num_epochs=20,
        binary_num_epochs=10
    )

run_experiment(config=military_vehicles_config)
```