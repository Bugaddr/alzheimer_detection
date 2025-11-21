#!/bin/bash
curl -L -o best-alzheimer-mri-dataset-99-accuracy.zip https://www.kaggle.com/api/v1/datasets/download/lukechugh/best-alzheimer-mri-dataset-99-accuracy
unzip best-alzheimer-mri-dataset-99-accuracy.zip
python3 -m venv ./venv
source ./venv/bin/activate
uv pip install tensorflow[and-cuda] numpy scikit-learn matplotlib seaborn
