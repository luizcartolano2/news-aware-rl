""" This module defines constants for the project, including paths to data directories.

"""
import os

# Define the root path of the project
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
# Define the path to the data directory
DATA_PATH = os.path.join(ROOT_PATH, "data")
# Define the path to the models directory
MODELS_PATH = os.path.join(ROOT_PATH, "models")
