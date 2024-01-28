<p align="center">
    <img alt="Logo" src="https://github.com/lab-v2/metacognitive_error_detection_and_correction_v2/blob/maintain_github/images/logo-transparent-png.png" width="440" height="500"/>
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

To demonstrate the use of the package, we provide a few running examples.

```python
if __name__ == '__main__':
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

# Acknowledgments

This research was supported in part by ...

This research was also supported by ...
