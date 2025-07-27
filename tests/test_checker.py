from pypso.util import checker


def test_is_converged():
    assert checker.is_converged(values=[1, 1, 1], consecutive=3) is True
    assert checker.is_converged(values=[1, 1, 2], consecutive=3) is False
    assert checker.is_converged(values=[1, 1, 0], consecutive=3) is False
    assert checker.is_converged(values=[1, 2, 1], consecutive=3) is False
    assert checker.is_converged(values=[1, 2, 3], consecutive=3) is False
