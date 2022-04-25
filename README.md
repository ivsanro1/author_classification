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


# How to run prediction
Assuming your python3.7 installation is callable via `python3`, you will be able to run the predictions by just running these lines in console:
```
pip install requirements.txt
python3 -m spacy download en_core_web_sm
python3 predict.py --input TO_BE_DEFINED --output TO_BE_DEFINED2
```

Running this command will read the input dataset in the file specified by `--input INPUT`. It will write the prediction output in the file specified by `--output OUTPUT`.


TODO: 
Add in docker all the project
Try to run inside docker