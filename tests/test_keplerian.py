from matplotlib import pyplot as plt
import pytest
import numpy as np

def test_import():
    from kima import keplerian
    from kima import kepler

def test_keplerian():
    from kima import keplerian, kepler

    t = np.linspace(0, 1, 10)
    P, K, ecc, w = 1.0, 1.0, 0.0, 0.0

    assert np.isfinite(keplerian([0.0], P, K, ecc, w, 0.0, 0.0))
    assert np.isfinite(keplerian(t, P, K, ecc, w, 0.0, 0.0)).all()
    np.testing.assert_allclose(
        keplerian([0.0, P/2, P], P, K, ecc, w, 0.0, 0.0),
        [K, -K, K]
    )

    ecc = 0.1
    np.testing.assert_allclose(
        keplerian([0.0, P/2, P], P, K, ecc, w, 0.0, 0.0),
        [K+ecc, -K+ecc, K+ecc]
    )

    t = np.sort(np.random.uniform(0, 1, size=100))
    for _ in range(10):
        P = np.random.uniform(2, 8)
        K = np.random.uniform(0.1, 10.0)
        ecc = np.random.uniform(0, 1)
        w = np.random.uniform(0, 2*np.pi)
        np.testing.assert_allclose(
            keplerian(t, P, K, ecc, w, 0.0, 0.0),
            kepler.keplerian2(t, P, K, ecc, w, 0.0, 0.0)
        )


@pytest.mark.skip(reason="only for testing")
def test_speed():
    import perfplot
    from kima import keplerian, kepler
    from radvel.kepler import rv_drive
    P, K, w, M0, M0_epoch = 1.0, 1.0, 0.0, 0.0, 0.0
    ecc = 0.2
    perfplot.show(
        #setup=lambda n: np.random.rand(n, n),  # or setup=np.random.rand
        setup=lambda n: np.linspace(0, 1, n),
        kernels=[
            lambda t: keplerian(t, P, K, ecc, w, M0, M0_epoch),
            lambda t: kepler.keplerian2(t, P, K, ecc, w, M0, M0_epoch),
            lambda t: rv_drive(t, [P, 0.0, ecc, w, K]),
        ],
        labels=['kima', 'kima array', 'radvel'],
        equality_check=None,
        n_range=[2**k for k in range(2, 10)],
    )
    plt.show()