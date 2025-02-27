# OmiDos

we introduce OmiDos, a novel dynamic orthogonal deep generative model, specifically designed to disentangle shared and unique molecular signatures across multi-omics layers in single-cell multi-omics data. Unlike prior methods that rely on simplistic assumptions about shared features, OmiDos leverages deep transfer learning to extract invariant shared signals from paired data while enforcing orthogonality constraints to separate modality-specific signals into distinct latent spaces. To address unpaired data, OmiDos incorporates an adversarial discriminator combined with noise contrastive estimation. Furthermore, OmiDos features a decoder layer enhanced by a maximum mean discrepancy regularization, enabling robust batch correction across diverse experimental conditions.

## Installation

### Dependencies

OmiDos requires the following:

- python (>= 3.8)

### User installation

The easiest way to install OmiDos is using pip.

python setup.py install

The documentation includes more detailed install instrcutions. OmiDos is free, software; you can redistribute it and/or modify it under the terms of the licence.
