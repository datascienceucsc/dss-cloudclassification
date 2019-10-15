import sys
import os
import glob

import click

# Load third party modules for environment testing
import numpy as np
import pandas as pd
import tensorflow as tf

from dss_kaggle_clouds.features.utils import verify_files_exist

@click.command()
@click.argument("RAW_DATA_DIR", type=click.Path(exists=True))
def run(raw_data_dir):
    """Command Line Interface
    
    The RAW_DATA_DIR is the data/ directory that includes the data/train_images or data/train.csv files
    should be passed on command line like:
    python3 run.py data/
    """
    if verify_files_exist(raw_data_dir):
        click.echo("Data dir is correctly built.")
    csv_file = os.path.join(raw_data_dir, "train.csv")
    train_images_dir = os.path.join(raw_data_dir, "train_images")
    image_files = list(map(os.path.abspath, glob.glob(os.path.join(train_images_dir, "*.jpg"))))
    
    click.echo("CSV filepath: {}".format(csv_file))
    click.echo("Train Images Count: {}".format(len(image_files)))

    sys.exit(0)


if __name__ == "__main__":
    run()
