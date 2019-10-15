# dss-cloudclassification
Repository for our work on the Understanding Clouds from Satellite Data competition on Kaggle

Your work on this project should follow our [Contribution guidelines](https://github.com/datascienceslugs/Useful-Documents/blob/master/CONTRIBUTING.md).

`dss_kaggle_clouds/models`, obviously includes the models, and `dss_kaggle_clouds/features` has any auxillary features/helper functions, to clean data to remove the bars from stitched images, or to do image augmentation to increase the size of training data.

This uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/) to manage a vitual environment, and we're limiting ourself to 3.6.2 to run on Hummingbird. The [requirements.txt](./requirements.txt) file is created from the [Pipfile](./Pipfile) and specifies locked dependencies for `setup.py`, to allow reproducable biulds.


## Installation:

If when you type `python -V`, you see `Python 3.x.x` (some version of 3), you can replace `python3` with `python` for the installation instructions.

Install pipenv, or skip this step if you already have pipenv installed.

```
python3 -m pip install --user pipenv
# or
python -m pip install --user pipenv
# or
pip install --user pipenv
```

If you can type `pipenv` in your shell and you it doesn't complain about that not being a command, you can replace `python3 -m pipenv` with just `pipenv`, else you can invoke it with `python3 -m pipenv` as a module


Make sure you're in the directory with the `Pipfile`, and then:

Install packages:

```
python3 -m pipenv install --skip-lock
```

If pipenv complains about you not having a python version 3.6 installed, you can install another version of python with [pyenv](https://github.com/pyenv/pyenv#installation). (`brew install pyenv` on mac)

And then you can install and specify the python version like:

```
pyenv install 3.6.2
python3 -m pipenv --python ~/.pyenv/versions/3.6.2/bin/python3
```

Enter the virtual environment

```
python3 -m pipenv shell

```

Install our code as a module in the virtualenv:

```
python3 setup.py install
```

Run the script:

```
python3 run.py --help
```

If you ever change any code in the `dss_kaggle_clouds` module/directory (make sure you're in the virtual environment), run:

```
python3 setup.py install  # to reinstall the package
# and then
python3 run.py data/raw # ... run the script
```

You can type `exit` to leave the virtual environment. (And `python3 -m pipenv shell` to re-enter)
