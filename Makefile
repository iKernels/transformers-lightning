clean:
	rm -rf dist/ build/ transformers_lightning.egg-info/ lightning_logs/ checkpoints/

compile:
	python setup.py sdist bdist_wheel
	twine check dist/*
