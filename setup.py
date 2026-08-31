from setuptools import find_packages, setup

setup(
    name="pressomancy",
    version="0.1.0",
    author="Deniz Mostarac",
    author_email="deniz.mostarac@uniroma1.it",
    description="Simulation package wrapping Espresso objects",
    packages=find_packages(exclude=["test", "test.*", "samples", "samples.*"]),
    test_suite="test",
    include_package_data=True,
    package_data={"pressomancy": ["resources/*.txt", "resources/*.py"]},
    install_requires=[
        "numpy", 
         "h5py",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change this if needed
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify the Python version requirement
)
