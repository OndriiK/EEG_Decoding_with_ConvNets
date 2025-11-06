# run_bcic.py
"""
Entry point for training DeepConvNet (and other models) on the
BCI Competition IV 2a (BNCI2014_001) dataset.
"""

import core
import dataset_bcic as ds

if __name__ == "__main__":
    core.run_all(ds, max_epochs=800)
