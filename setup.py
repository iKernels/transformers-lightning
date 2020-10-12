from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='transformers_lightning',
    version='0.1',
    description='Easily deploy Transformers models over Lightning',
    long_description=long_description,
    url='git@github.com:lucadiliello/transformers-lightning.git',
    author='Luca Di Liello',
    author_email='luca.diliello@unitn.it',
    license='GNU v2',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU v2 License",
        "Operating System :: OS Independent",
    ]
)