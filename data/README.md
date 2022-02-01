

Make sure to have the dataset directories outside the root folder of this directory.

Use python 3.7 version to create the virtual environment.

    # to create venv:
    sudo virtualenv venv

    # to enter venv:
    source venv/bin/activate

    # to exit venv:
    deactivate

When activated the virtual environment, install the dependencies.

    pip install -r requirements.txt

Create a kernel for the virtual environment for the jupyter notebook.

    sudo python3 -m ipykernel install --name=vader_venv
