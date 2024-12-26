# Importing core functionalities for easier access from the main package
from .trainer import train
from .model import BertForMLM, BertForDownstream

# Setting a list of all available components for cleaner imports
__all__ = [
    "train",
    "BertForMLM", "BertForDownstream"
]
