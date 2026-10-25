from src.monitor.drift import compute_psi

def test_psi():
    assert compute_psi([1, 2, 3], [1, 2, 3]) >= 0
