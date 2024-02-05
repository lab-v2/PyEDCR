<p align="center">
    <img alt="Logo" src="images/logo-transparent-png.png" width="440" height="500"/>
</p>

<!-- [![pages-build-deployment](https://github.com/krichelj/PyDiffGame/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/krichelj/PyDiffGame/actions/workflows/pages/pages-build-deployment) -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

- [What is this?](#what-is-this)
- [Local Installation](#local-installation)
- [Tutorial](#tutorial)
- [Acknowledgments](#acknowledgments)

# What is this?

[`PyEDCR`](https://github.com/lab-v2/metacognitive_error_detection_and_correction_v2/tree/master) is a Python implementation of Error Detection and Correction Rules. The goal of PyEDCR is to use a set of conditions to learn when a machine learning model makes an incorrect prediction and to fix it to the correct prediction. The rules used in this package were constructed in this article:

- https://arxiv.org/pdf/2308.14250.pdf

# Installation

To install this package run this from the command prompt:

```
pip install PyEDCR
```

The package was tested for Python >= 3.10, along with the listed packages versions in [`requirements.txt`](https://github.com/lab-v2/metacognitive_error_detection_and_correction_v2/blob/maintain_github/requirements.txt)

# Tutorial

To demonstrate the use of the package, we provide running examples. The main function of this package is run_EDCR_pipeline from the EDCR_pipeline module.

```python
from PyEDCR.EDCR_pipeline import run_EDCR_pipeline

combined = False
conditions_from_main = True
print(utils.red_text(f'\nconditions_from_secondary={not conditions_from_main}, '
                     f'conditions_from_main={conditions_from_main}\n' +
                     f'combined={combined}\n' + '#' * 100 + '\n'))

run_EDCR_pipeline(main_lr=0.0001,
                  combined=combined,
                  loss='soft_marginal',
                  conditions_from_secondary=not conditions_from_main,
                  conditions_from_main=conditions_from_main,
                  consistency_constraints=True,
                  multiprocessing=True)
```

Here, 'main_lr' refers to the learning rate of the model in question. 'combined' is a flag for individual vs combined models. Combined models predict both fine and coarse grain while individual predicts one or the other. 'loss' refers to the specified loss. In our development, we used soft_marginal and BCE. 'conditions_from_main' specifies where the conditions for EDCR come from. If 'conditions_from_main' is true, a combined model will use it's own predictions as conditions for EDCR. If false, conditions will be from another model. 'consistency_constraints' is a flag to print the information for the recovered constraints and the mean constraints among all fine and coarse classes. 'multiprocessing' is used to enable multiprocessing.

# Acknowledgments

This research was supported in part by ...

This research was also supported by ...
