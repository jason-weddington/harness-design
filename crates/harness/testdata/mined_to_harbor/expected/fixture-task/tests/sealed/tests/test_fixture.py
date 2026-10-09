from fixture.widget import Widget


class TestWidget:
    def test_existing_behaviour(self):
        assert Widget([]).count() == 0

    def test_new_behaviour(self):
        assert Widget([1, 2]).count() == 4
