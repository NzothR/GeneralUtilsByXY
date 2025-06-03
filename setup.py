from setuptools import setup, find_packages

setup(
    name="GeneralUtilsByXY",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python-headless",
        "matplotlib",
        "pytest",
        "numpy==1.26.4",
        "torch==2.7.0",
        "torchvision==0.22.0",
        "tqdm==4.67.1"
    ],
    python_requires='>=3.10'
)