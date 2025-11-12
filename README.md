# ICOS Intelligence Coordination API

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/icos-project/intelligence-module)
[![ICOS Project](https://img.shields.io/badge/ICOS-Project-orange.svg)](https://www.icos-project.eu/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3721889.3721929-blue.svg)](https://doi.org/10.1145/3721889.3721929)

The AI coordination module facilitates optimisation, predictive analytics, and applying machine learning models across the edge-cloud continuum. It provides coordination between the meta-kernel and user layers, enabling services and collaborative learning capabilities across the ICOS continuum.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Documentation](#documentation-index)
- [Getting Started](#starting-the-api-service)
- [Citation](#citation)
- [Credits](#credits)
- [License](#license)

## Overview

The ICOS Intelligence Layer Coordination API offers mechanisms for:
- **Resource prediction**: Estimating CPU/Memory consumption of ICOS agents
- **Model drift detection**: Monitoring and detecting model performance degradation
- **Explainable AI**: Improving AI trustworthiness through model prediction explanations
- **Model management**: Performance monitoring, compression, and scoring
- **Federated Learning**: Privacy-preserving collaborative learning for trust-sensitive applications

## Key Features

- **Predictive Analytics**: CPU usage prediction models for resource optimization
- **Trustworthy AI (TAI)**: Drift detection, model explainability, and monitoring capabilities
- **Federated Learning**: Distributed learning across edge-cloud continuum
- **Model Compression**: Efficient model deployment for edge devices
- **Multiple ML Frameworks**: Support for PyTorch, XGBoost, ARIMA, and LSTM models
- **RESTful API**: Easy integration with existing systems

## Credits

This open-source AI framework has been developed by:
- **CeADAR** (Ireland's National Centre for Applied AI, University College Dublin): Jaydeep Samanta, Sebastian Cajas Ordoñez, Romila Ghosh, Dr. Andrés L. Suárez-Cetrulo, and Dr. Ricardo Simón Carbajo
- **National and Kapodistrian University of Athens (NKUA)**: Anomaly detection and LSTM models

This work has been performed as part of the Intelligence layer of the EU HORIZON project No. 101070177 - IoT to Cloud Operating System (ICOS).

## Documentation Index

Below is a list of related documentation sections to help you navigate the Intelligence Layer and its capabilities:

1. [**API Documentation**](api.md)  
2. [**Backend Services**](backend.md)
3. [**Deployment Guide**](deployment.md)  
4. [**Usage**](usage.md)  
5. [**Development & Contribution**](development.md)  

## License

- This project, developed by CeADAR Ireland, is licensed under the Apache License v2.0 (Apache-2.0) (https://www.apache.org/licenses/LICENSE-2.0). This includes the source code of both the API and models contained.
- Energy consumption prediction, Federeated Learning, LSTM models and classification models for anomaly detection provided by the National and Kapodistrian University of Athens (NKUA) are included in the repository. However, their source code is not included. The licensing terms of NKUA's source code, used to generate these runnable files, are not distributed with this project and may or may not be similar to GPLv3.

## Dependencies

This project uses the following external dependencies:

* `DataClay` (BSD 3-Clause "New" or "Revised" License, https://pypi.org/project/dataclay/)

Please note that `DataClay` is not included in this repository; it is only imported as an external dependency.


## Architecture

The Intelligence Coordination API implements policies for utilising, sharing, and updating machine learning models across the edge-cloud continuum. It acts as an interface coordinating between the meta-kernel and user layers, enabling ICOS to learn collaboratively with other intelligence layers in the same domain.

## Files and Folder structure

```
├── .gitignore  
├── bentofile.yaml   
├── LICENSE
├── README.md         
├── oasis
│   ├── analytics
│   │   ├── dataframes.py
│   │   ├── lstm_model.py
│   │   ├── metrics.py
│   │   └── model_metrics.py
│   ├── dataset
│   ├── api_service.py
│   ├── api_service_configs.json
│   ├── api_train.py
│   ├── bentofile.yaml
│   ├── clean_dockers.sh
│   ├── configuration.yaml
│   ├── requirements.txt
│   ├── models
│   │   ├── arima
│   │   │   └── arima_compiler.py
│   │   ├── management
│   │   │   └──ai_model_repo.py
│   │   │   └── registry.py
│   │   ├── pytorch
│   │   │   └── pytorch_compiler.py
│   │   └── xgboost
│   │       └── xgboost_compiler.py
│   ├── processing
│   │   ├── process.py
│   │   └── utils.py
│   └── tai
│       ├── model_explainability.py
│       ├── monitoring.py
│       ├── model_sync.py
│       └── organizer_fl.py   
├── test
│   ├── api
    │    ├── misc_services.py
    │    ├── model_inference.py
    │    ├── model_training.py
    ├── marketplace
        ├── test_hf.py
        ├── test_libraries.py
```
* .gitignore:
    -  Files that should be ignored by git. Add seperate .gitignore files in sub folders if needed

* bentofile.yaml:
    - Configuration file to create a bento.

* oasis/src:
    - This folder contains the source code for running the API
    1. dataset:
        - This folder will contain the dataset that will be used to train the cpu utilisation models
        - An init file for initialising the dataset using a mapper function
    2. api_train_model.py:
        - This Python module encompasses the necessary functionalities for training ML algorithms. These functions handle crucial tasks such as data ingestion, data preprocessing, feature selection, and data splitting, ensuring that the input training and testing data are appropriately formatted for the training model. Once the data is preprocessed, it can be passed as input to the training function, which executes the model training and generates model metrics.
        - Furthermore, these trained models are saved using BentoML along with custom objects that may be utilised during the model prediction.
    3. api_service.py:
        - This is an API service module that provides API endpoints for training and inference using trained models.
        - This module calls the api_train_model.py file internally to retrieve trained models or construct new ones.
        - The predict function, which receives the best-trained model, plays a crucial role in returning the predicted values to their original format. This function can accept various input parame ters such as data input, trained model, and preprocessors, if required, and ultimately delivers the results.
    4. api_service_configs.json:
        - This is a configuration files that is used to store default parameters.
        - This may be changed by a user by sending POST request through the API endpoints.
    5. requirements.txt:
        - This contains all necessary libraries along with versions required for the API

* test: 
    - Unit Test Cases for the ICOS Intelligence API.

## Starting the API Service

### Prerequisites

- Docker installed
- Minimum 40GB RAM recommended
- x86_64 architecture (for the alpha version)

### Docker Deployment

A pre-built Docker image is available [here](https://drive.google.com/file/d/1dIpbUn8GnhN93AHQLLA4vW6GZNMWuvsq/view?usp=sharing).

**Launch the service:**

```bash
docker run --network host -it --rm \
  -p 3000:3000 -p 5000:5000 -p 8888:8888 \
  --cpus 15 --memory 20g \
  -e BENTOML_CONFIG_OPTIONS='api_server.traffic.timeout=1200 runners.resources.cpu=0.5 runners.resources."nvidia.com/gpu"=0' \
  analytics:latest serve
```

**Port Configuration:**
- `3000`: API service endpoint
- `5000`: MLflow UI service
- `8888`: Jupyter Lab service (if supported)

**Resource Configuration:**
- `--cpus 15`: Number of CPU cores (15 ML models being served in parallel)
- `--memory 20g`: Minimum memory allocation for full functionality

**BentoML Configuration:**
- `api_server.traffic.timeout=1200`: API traffic timeout (seconds)
- `runners.resources.cpu=0.5`: CPU resources per runner instance
- `runners.resources."nvidia.com/gpu"=0`: GPU resources (set to 0 for CPU-only)

Once started, the API will be accessible at `http://0.0.0.0:3000`.


## Integrating New Models

### 1. Update Dataset

Add your dataset to the `dataset/` directory and update the initialization file to map the relevant dataset.

### 2. Model Training

1. Create a function definition inside the `ModelTrain` class in [api_train_model.py](oasis/api_train.py)
2. Call your function through the `initiate_train` method
3. The trained model and associated scalar objects will be saved to the BentoML repository

### 3. API Service & Model Inference

Once saved to the BentoML repository, your model can be used for inference through the BentoML service API endpoints defined in [api_service.py](oasis/api_service.py).

## Tested Platforms

The API has been successfully tested on the following hardware configurations:

### 1. ARM64 Orange Pi Board
- **CPU:** Rockchip RK3588 8-cores (Cortex-A76, Cortex-A55)
- **OS:** Linux orangepi5 5.10.110 (Debian)

### 2. NVIDIA Jetson AGX Orin
- **CPU:** NVIDIA Carmel CPU Arm 12-core Cortex-A78AE
- **GPU:** Ampere GPU (2048 CUDA cores, 64 Tensor cores)
- **DLA:** 2x NVIDIA Deep Learning Accelerator
- **OS:** JetPack 5.1 (Ubuntu 20.04)

### 3. x86_64 AMD Threadripper Server
- **CPU:** AMD Ryzen Threadripper PRO 5975WX 32-Cores
- **GPU:** NVIDIA RTX 4090
- **OS:** Ubuntu 22.04

### 4. x86_64 AMD Server (NCSRD Infrastructure)
- **CPU:** AMD Opteron 240 (Gen 1)
- **OS:** Ubuntu 20.04.6 Server
- **Virtualization:** Proxmox
- **VM Configuration:** [Available here](https://newrepository.atosresearch.eu/index.php/f/1271920) (WP5 folder)

## System Requirements

- **Minimum RAM:** 40GB (containerized version requires 20GB minimum)
- **Storage:** Sufficient space for Docker images and model storage
- **Architecture:** x86_64 (alpha version)


---


### Acknowledgements

This work has been performed as part of the Intelligence layer of the EU HORIZON project No. 101070177 - IoT to Cloud Operating System (ICOS).

🇪🇺 This work has received funding from the European Union's HORIZON research and innovation programme under grant agreement No. 101070177.


## Legal

**Copyright © 2022-2025 CeADAR Ireland. All rights reserved.**

The ICOS Coordination API is released under the Apache License v2.0. See [LICENSE](LICENSE) for details.

🇪🇺 This work has received funding from the European Union's HORIZON research and innovation programme under grant agreement No. 101070177.


## Citation

If you use this software in your research, please cite:

```bibtex
@inproceedings{ICOS-paper,
  title = {{ICOS An Intelligent MetaOS for the Continuum}},
  author = {Garcia, Jordi and Masip-Bruin, Xavi and Giannopoulos, Anastasios and Trakadas, Panagiotis and Cajas Ordoñez, Sebastián A. and Samanta, Jaydeep and Suárez-Cetrulo, Andrés L. and Simón Carbajo, Ricardo and Michalke, Marc and Admela, Jukan and Jaworski, Artur and Kotliński, Marcin and Giammatteo, Gabriele and D'Andria, Francesco},
  year = {2025},
  isbn = {9798400715600},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3721889.3721929},
  doi = {10.1145/3721889.3721929},
  booktitle = {Proceedings of the 2nd International Workshop on MetaOS for the Cloud-Edge-IoT Continuum},
  pages = {53–59},
  numpages = {7},
  location = {Rotterdam, Netherlands},
  series = {MECC '25}
}
```

