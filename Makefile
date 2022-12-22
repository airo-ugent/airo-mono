SHELL=/bin/bash

pytest:
	python3 -m pytest --cov-report html --cov-report term --cov=. -v --color=yes -m "not expensive"
