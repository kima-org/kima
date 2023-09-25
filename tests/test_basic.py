import kima

def test_glob():
    assert kima.glob('*.md') == ['README.md']

def test_extensions_exist():
    kima.RVData
    kima.RVmodel
    kima.GPmodel
    kima.RVFWHMmodel
    kima.run

def test_api():
    D = kima.RVData('tests/simulated1.txt')
    # print(D.N)
    m = kima.RVmodel(True, 0, D)
    m.trend = True
    m.degree = 2
    # print(m.trend)
    # print(help(kima.run))

def test_RVData():
    D = kima.RVData('tests/simulated1.txt')
    D = kima.RVData(['tests/simulated1.txt', 'tests/simulated2.txt'])

def test_RVmodel():
    m = kima.RVmodel(True, 0, kima.RVData('tests/simulated1.txt'))

def test_GPmodel():
    m = kima.GPmodel(True, 0, kima.RVData('tests/simulated1.txt'))

def test_RVFWHMmodel():
    m = kima.RVFWHMmodel(True, 0, kima.RVData('tests/simulated1.txt'))


def test_distributions():
    from kima import distributions
    from kima.distributions import Gaussian, Uniform
    u = Uniform()
    # print(u)