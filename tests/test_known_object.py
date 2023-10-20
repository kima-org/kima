import pytest
import kima
from kima import RVData, RVmodel
from kima.distributions import Uniform


def test_get_KO():
    D = RVData('tests/simulated1.txt')
    m = RVmodel(True, 0, D)
    
    assert not m.known_object, 'm.known_object should be False'
    assert m.n_known_object == 0, 'm.n_known_object should be 0'
    assert len(m.KO_Pprior) == 0, 'm.KO_Pprior should be empty'

def test_set_KO():
    D = RVData('tests/simulated1.txt')
    m = RVmodel(True, 0, D)

    m.set_known_object(2)
    assert m.known_object, 'm.known_object should be True'
    assert m.n_known_object == 2, 'm.n_known_object should be 2'
    assert len(m.KO_Pprior) == 2, 'm.KO_Pprior should have length 2'
    assert m.KO_Pprior[0] is None, 'm.KO_Pprior should all be None'
    assert m.KO_Pprior[1] is None, 'm.KO_Pprior should all be None'

    m.KO_Pprior = [Uniform(1,2), Uniform(2,3)]

    with pytest.raises(RuntimeError):
        kima.run(m, steps=10)