__all__ = (
    'GAIA', 'GAIA_free', 'RVGAIA',
    'Gaia_astrometry_BH3_datafile', 'Gaia_RVs_BH3_datafile'
)

from .GAIA import GAIA
from .GAIA_free import GAIA_free
from .RVGAIA import RVGAIA

from pathlib import Path

here = Path(__file__).parent

Gaia_astrometry_BH3_datafile = (here / "Gaia_astrometry_BH3.gaia").as_posix()
Gaia_RVs_BH3_datafile = (here / "Gaia_RVs_BH3.rdb").as_posix()