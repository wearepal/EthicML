# content of test_class.py
class TestClass(object):
    def test_one(self):
        x: str = "this"
        assert 'h' in x

    def test_two(self):
        x: str = "hello"
        assert hasattr(x, 'casefold')

    def test_three(self):
        x: str = "hello"
        assert hasattr(x, 'capitalize')