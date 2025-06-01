import pandas as pd
from importlib.resources import files


def load_simple_shop():
    """Load a built-in dataset from the package's /data directory."""
    try:
        csv_path = str(files("product_sense.datasets").joinpath("simple_shop.csv"))
        return pd.read_csv(csv_path)
    except Exception as e:
        raise IOError(f"Could not load SimpleShop dataset: {str(e)}")
