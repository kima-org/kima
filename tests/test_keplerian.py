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

def test_keplerian_is_array():
    from kima import keplerian

    t = np.linspace(0, 1, 10)
    P, K, ecc, w, M0, M0_epoch = 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
    v = keplerian(t, P, K, ecc, w, M0, M0_epoch)
    assert isinstance(v, np.ndarray)


def test_keplerian_etv():
    from kima.kepler import keplerian_etv

    epochs = np.linspace(0, 1, 10)
    P, K, ecc, w, eph1 = 1.0, 1.0, 0.0, np.pi/2, 0.01
    
    assert np.isfinite(keplerian_etv([0.0], P, K, ecc, w, 0.0, eph1))
    assert np.isfinite(keplerian_etv(epochs, P, K, ecc, w, 0.0, eph1)).all()
    np.testing.assert_allclose(
        keplerian_etv([0.0, P/2/eph1, P/eph1], P, K, ecc, w, 0.0, eph1),
        [K, -K, K]
    )

    ecc = 0.1
    cnu1 = -0.19738041725338823
    snu1 = 0.9803269714155979
    np.testing.assert_allclose(
        keplerian_etv([0.0, P/4/eph1, P/2/eph1], P, K, ecc, w, 0.0, eph1),
        [K, K*((1-ecc**2)/(1+ecc*cnu1)*cnu1+ecc), -K]
    )
    
    w=0
    np.testing.assert_allclose(
        keplerian_etv([0.0, P/4/eph1, P/2/eph1], P, K, ecc, w, 0.0, eph1),
        [0, K*((1-ecc**2)**0.5)/(1+ecc*cnu1)*snu1, 0]
    )

def test_keplerian_gaia():
    from kima.kepler import keplerian_gaia

    t = np.linspace(0, 1, 10)
    psi = np.linspace(0,2*np.pi,10)
    P, ecc, A, B, F, G = 1.0, 0.0, 1.0, -1.0, 1.0, -1.0
    
    assert np.isfinite(keplerian_gaia([0.0],[0.0],A,B,F,G,ecc,P,0.0,0.0))
    assert np.isfinite(keplerian_gaia(t,psi,A,B,F,G,ecc,P,0.0,0.0)).all()


    ecc=0.1
    
    np.testing.assert_allclose(
        keplerian_gaia([0.0,P/4 - ecc*P/2/np.pi,P/2],[0.0,0.0,np.pi/2],A,B,F,G,ecc,P,0.0,0.0),
        [1-ecc, (1-ecc**2)**(0.5) - ecc, 1+ecc]
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