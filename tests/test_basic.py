import kima as m

def test_glob():
    assert m.glob('*.md') == ['README.md']
