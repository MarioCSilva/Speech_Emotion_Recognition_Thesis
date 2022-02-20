Make sure to have the dataset directories outside the root folder of this directory.

## Extract Features from Audio Files

Use python 3.7 version to create the virtual environment.

    # to create venv:
    sudo virtualenv venv

    # to enter venv:
    source venv/bin/activate

    # to exit venv:
    deactivate

When activated the virtual environment, install the dependencies.

    pip install -r requirements.txt

Run feature_extraction.py.

    python .\feature_extraction.py


## Analysis of Features

Notebook feature_analysis.ipynb