# dss-cloudclassification
Repository for our work on the Understanding Clouds from Satellite Data competition on Kaggle

Your work on this project should follow our [Contribution guidelines](https://github.com/datascienceslugs/Useful-Documents/blob/master/CONTRIBUTING.md).

`dss_kaggle_clouds/models`, obviously includes the models, and `dss_kaggle_clouds/features` has any auxillary features/helper functions, to clean data to remove the bars from stitched images, or to do image augmentation to increase the size of training data.

This uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/) to manage a vitual environment, and we're limiting ourself to 3.6.2 to run on Hummingbird. Using `pipenv run pip freeze`, one can create a [requirements.txt](./requirements.txt) file from the [Pipfile](./Pipfile) that can be used to install locked dependencies on other systems/servers.

To install stuff and be in virtual environment.

```
## change so that you're in the directory# install pipenv or install it with pip-python
python3 -m pip install --user pipenv
# install packages
python3 -m pipenv install --skip-lock
# enter the virtualenv
python3 -m pipenv shell
# install the python package
python3 setup.py install
# run the script to test
kaggle_clouds
```
