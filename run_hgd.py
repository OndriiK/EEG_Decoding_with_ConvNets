# run_hgd.py
"""
Entry point for training DeepConvNet (and other models) on the
High Gamma Dataset (HGD / Schirrmeister2017).
"""

import core
import dataset_hgd as ds

if __name__ == "__main__":
    core.run_all(ds, max_epochs=800)
