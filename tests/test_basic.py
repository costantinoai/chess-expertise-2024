import os

def test_readme_exists():
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'README.md'))
