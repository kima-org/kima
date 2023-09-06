import kima

def test_glob():
    assert kima.glob('*.md') == ['README.md']

def test_extensions_exist():
    kima.RVData
    kima.RVmodel
    kima.run

def test_api():
    D = kima.RVData('tests/simulated1.txt')
    # print(D.N)
    m = kima.RVmodel(True, 0, D)
    m.trend = True
    m.degree = 2
    # print(m.trend)
    print(help(kima.run))