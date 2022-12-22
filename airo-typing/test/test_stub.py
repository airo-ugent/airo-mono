def test_stub():
    # each package should have at least one test to make the CI pipeline happy
    # pytest returns an error if no tests are found
    # cf https://github.com/pytest-dev/pytest/issues/2393
    assert True
