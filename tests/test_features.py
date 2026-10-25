from src.features.build import get_preprocessor

def test_preprocessor_init():
    prep = get_preprocessor()
    assert prep is not None
