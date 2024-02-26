from setuptools import setup, find_packages

setup(
    name="epicare",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gym==0.23.1",
        "numpy",
        "pandas",
        "h5py",
        "pyrallis",
        "torch",
        "tqdm",
        "wandb",
        "scipy",
    ],
)
