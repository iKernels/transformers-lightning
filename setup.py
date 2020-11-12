import setuptools
import json

def load_long_description():
    with open("README.md", "r") as fh:
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

def get_version():
    # get semver version [major.minor.patch]
    json_version = {}
    with open('.version.json', 'r') as f:
        json_version = json.load(f)
    return '.'.join(str(w) for w in [json_version['major'],json_version['minor'],json_version['patch']])

setuptools.setup(
    name='transformers_lightning',
    version=get_version(),
    description='Easily deploy Transformers models over Lightning',
    long_description=load_long_description(),
    url='git@github.com:lucadiliello/transformers-lightning.git',
    author='Luca Di Liello',
    author_email='luca.diliello@unitn.it',
    license='GNU v2',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU v2 License",
        "Operating System :: OS Independent",
    ]
)