SHELL=/bin/bash

pytest:
	python3 -m pytest --cov-report html --cov-report term --cov=airo_core/ -v --color=yes -m "not expensive"

mypy:
	python -m mypy airo_core/