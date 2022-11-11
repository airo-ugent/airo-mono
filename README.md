# airo_core

## goals
- create base classes / interfaces for abstracting (simulated) hardware to allow for interchanging different simulators/ hardware on the one hand and different 'decision making frameworks' on the other hand.
- provide functionality that is required to quickly create robot behavior à la Folding competition @ IROS22
- facilitate research by providing, wrapping or linking to common operations for Robotic Perception/control

## Developer guide
### Coding style and testing
- formatting with black
- linting with flake8
- above is enforced w/ pre-commit.
- typing with mypy?
- docstrings in reST (Sphinx) format ([most used](https://stackoverflow.com/questions/3898572/what-are-the-most-common-python-docstring-formats))
- testing with pytest. All tests are grouped in `/test`


### Design
- attributes that require complex getter/setter behaviour should use python properties
- the easiest code to maintain is no code -> does it already exists somewhere?