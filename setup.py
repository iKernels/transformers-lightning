import setuptools

from transformers_lightning.info import __version__


def load_long_description():
    with open("DESCRIPTION.md", "r") as fh:
        long_description = fh.read()
    return long_description


def load_requirements():
    requirements = []
    with open("requirements.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                requirements.append(line)
    return requirements


setuptools.setup(
    name='transformers_lightning',
    version=__version__,
    description='Easily deploy Transformers models over Lightning',
    long_description=load_long_description(),
    long_description_content_type="text/markdown",
    url='https://github.com/lucadiliello/transformers-lightning.git',
    author='Luca Di Liello',
    author_email='luca.diliello@unitn.it',
    license='GNU v2',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Natural Language :: English"
    ]
)
