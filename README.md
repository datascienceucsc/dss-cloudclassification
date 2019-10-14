# dss-cloudclassification
Repository for our work on the Understanding Clouds from Satellite Data competition on Kaggle

Your work on this project should follow our [Contribution guidelines](https://github.com/datascienceslugs/Useful-Documents/blob/master/CONTRIBUTING.md).

`dss_kaggle_clouds/models`, obviously includes the models, and `dss_kaggle_clouds/features` has any auxillary features/helper functions, to clean data to remove the bars from stitched images, or to do image augmentation to increase the size of training data.

This uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/) to manage a vitual environment, and we're limiting ourself to 3.6.2 to run on Hummingbird. Using `pipenv run pip freeze`, one can create a [requirements.txt](./requirements.txt) file from the [Pipfile](./Pipfile) that can be used to install locked dependencies on other systems/servers.


