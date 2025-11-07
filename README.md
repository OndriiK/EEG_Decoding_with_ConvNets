[![Python][python-shield]][python-url]
[![PyTorch][pytorch-shield]][pytorch-url]
[![MOABB][moabb-shield]][moabb-url]
[!License][license-shield]][license-url]

---

# EEG_Decoding_with_ConvNets

## Project Overview
This project focuses on reproducing results from an academic study exploring the applications of convolutional neural networks in EEG decoding.

##### Academic Study:
> **Schirrmeister, R. T., Springenber, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017)**
> *Deep learning with convolutional neural networks for EEG decoding and visualization.*
> *Human Brain Mapping, 38*(11), 5391-5420.
> DOI: [10.1002/hbm.23730](https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730)

We reproduced the evaluation results in Table 2 of the paper using the DeepConvNet architecture on three EEG datasets:
- **BCIC**: BCI Competition IV 2a implemented from MOABB
- **HGD**: High-Gamma Dataset implemented from MOABB
- **Combined**: BCIC + HGD

All our experiments use the [**MOABB framework**](https://github.com/NeuroTechX/moabb/tree/develop) for EEG benchmarking, and the [**Braindecode**](https://braindecode.org/) library for deep learning.

---

## Contributors
Abubakar Mohamed Said Adan, Ondrej Kozanyi, Miina Mäkinen, and Julia Trznadel.

Course: MSc Seminars in Data Science
University: IT University Copenhagen
Year: 2025


<!-- MARKDOWN LINKS & IMAGES -->
[python-shield]:https://img.shields.io/badge/python-3.9%2B-blue.svg
[pythin-url]:https://www.python.org/
[pytoech-shield]:https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg
[pytorch-url]:https://pytorch.org/
[moabb-shield]:https://img.shields.io/badge/MOABB-develop-green.svg
[moabb-url]:https://github.com/NeuroTechX/moabb
[license-shield]:https://img.shields.io/badge/license-MIT-lightgrey.svg
[license-url]:LICENSE
