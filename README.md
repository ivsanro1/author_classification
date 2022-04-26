# Author Classification for Texts (books)
This repository contains all the assets to perform Exploratory Data Analysis on data, Training, Evaluation and Prediction of a Machine Learning system that predicts authors given text fragments.

# Environment
We highly encourage you to run the docker-compose.yaml and go to `localhost:8890` to run the notebooks and the scripts to predict, given the number of dependencies of the project. However, if you are running on a Ubuntu 18.04 with Python 3.7 installed with a Jupyter installation, you can only install the dependencies in `requirements.txt` via `pip install requirements.txt` before running the notebooks and/or prediction script.

# Software requirements
Operative system: Linux (Ubuntu 18.04). Provided you have Docker installed, you can use the Dockerfile which will run on Ubuntu 18.04
Python 3.7. Provided you have Docker installed, you can use the Dockerfile which will have it installed

# Hardware requirements
Recommended RAM
- At least 6 GB to train the model.
- At least 4 GB to run predictions with the model. Take into account that it depends on how much data you try to predict, so consider splitting the prediction in batches if you get mem errors.

# Main repository assets
- `notebooks/exploratory_data_analysis.ipynb`: Notebook for Exploratory Data Analysis of the training data under `data/train.csv`
- `notebooks/train_model.ipynb`: Notebook to train the Author Classification model, and fitting other assets needed for feature extraction. Running this notebook will use the trainindata under `data/train.csv`, and validate the trained model with a split of that training set.
- `predict.py`: Script to infer using the trained models, which will be under `models/`. You can run the script without having trained the models yourself and the models will be automatically downloaded from Google Drive and unpacked in the directory `models/`.

# How to run the prediction (`predict.py`)
Assuming you have Docker installed, first run:

```
docker-compose up --build
```

This will install all the needed dependencies and run a Jupyter Notebook Server which will be accessible once the container is running if you navigate with your web browser to:

```
localhost:8890
```

You will see the Jupyter Notebook interface. Navigate to the top right corner and open a new terminal:

 ![](how_to_open_terminal_jupyter_notebook.gif)

Once in the terminal run this command to run the prediction:

```
python3 predict.py data/test.csv results.csv
```


The script will read the input dataset in the file specified by the first parameter and it will write the prediction output in the file specified by the second parameter.