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

To demonstrate the use of the package, consider the following running example using the 'simulate_for_values' function from NeuralPyEDCR.py

```python
import os
from NeuralPyEDCR import simulate_for_values

data_str = 'imagenet'
main_model_name = binary_model_name = 'dinov2_vits14'
secondary_model_name = 'dinov2_vitl14'
main_lr = secondary_lr = binary_lr = 0.000001
original_num_epochs = 8
secondary_num_epochs = 2
binary_num_epochs = 5
number_of_fine_classes = 42

binary_l_strs = list({f.split(f'e{binary_num_epochs - 1}_')[-1].replace('.npy', '')
                          for f in os.listdir('binary_results')
                          if f.startswith(f'{data_str}_{binary_model_name}')})

simulate_for_values(total_number_of_points=1,
                    min_value=0.1,
                    max_value=0.1,
                    binary_l_strs=binary_l_strs,
                    binary_lr=binary_lr,
                    binary_num_epochs=binary_num_epochs,
                    multi_processing=True,
                    secondary_model_name=secondary_model_name,
                    secondary_model_loss='BCE',
                    secondary_num_epochs=secondary_num_epochs,
                    secondary_lr=secondary_lr,
                    maximize_ratio=True,
                    lists_of_fine_labels_to_take_out=[[]],
                    negated_conditions=False)
```