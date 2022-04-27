# Author Classification for Texts (books)
This repository contains all the assets to perform Exploratory Data Analysis on data, Training, Evaluation and Prediction of a Machine Learning system that predicts authors given text fragments.

# Environment
We highly encourage you to run the docker-compose.yaml and go to `localhost:8890` to run the notebooks and the scripts to predict, given the number of dependencies of the project. This is explained below in the section `How to run the prediction`. However, if you are running on a Ubuntu 18.04 with Python 3.7 installed with a Jupyter installation, you can only install the dependencies in `requirements.txt` via `pip install requirements.txt` before running the notebooks and/or prediction script.

# Software requirements
Operative system: Linux (Ubuntu 18.04). Provided you have Docker installed, you can use the Dockerfile which will run on Ubuntu 18.04
Python 3.7. Provided you have Docker installed, you can use the Dockerfile which will have it installed

# Hardware requirements
Recommended RAM
- At least 6 GB to train the model.
- At least 4 GB to run predictions with the model. Take into account that it depends on how much data you try to predict, so consider splitting the prediction in batches if you get mem errors.

# Main repository assets
## Delivery
- Exploratory data analysis as a form of Jupyter Notebook: `notebooks/exploratory_data_analysis.ipynb`
- Standalone script that runs the whole pipeline on the test set and creates a file with the results: `predict.py` (see usage in next section). You can run the script without having trained the models yourself and the models will be automatically downloaded from Google Drive and unpacked in the directory `models/`.
- File containing the predictions for the test set: `results.csv`.
## Extra
- `notebooks/train_model.ipynb`: Notebook to train the Author Classification model, and fitting other assets needed for feature extraction. Running this notebook will use the trainindata under `data/train.csv`, and validate the trained model with a split of that training set.

# How to run the prediction (`predict.py`)
Assuming you have Docker installed, clone this repository, go to the root of it and run:

```
docker-compose up --build
```

This will install all the needed dependencies and run a container with a Jupyter Notebook Server which will be accessible once the container is up and running.

After it is up and running, navigate with your web browser to:

```
localhost:8890
```

You will see the Jupyter Notebook interface. Then, navigate to the top right corner and open a new terminal:

 ![](how_to_open_terminal_jupyter_notebook.gif)

Once in the terminal, run this command to run the prediction:

```
python3 predict.py data/test.csv results.csv
```


The script will read the input dataset in the file specified by the first parameter (`data/test.csv`) and it will write the prediction output in the file specified by the second parameter (`results.csv`).


# Code and documentation

The main code for this project is under the package `author_classification/`. This package is installable via `python3 setup.py install`, and it can be imported after being installed. You can see how this is being done in the Dockerfile.

The documentation for this module can be found under `doc/` and it has been generated with `pdoc3`.

To navigate the documentation, simply open `doc/author_classification/index.html` with your browser.