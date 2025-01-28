# what a beautiful hack!
import sys
from glob import glob
from pathlib import Path
import tempfile
import tarfile
import zipfile

def replace_tilde_in_targz(tar_gz):
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)

        # extract archive to temporry directory
        with tarfile.open(tar_gz) as r:
            r.extractall(td)

        pkg_info = next(tdp.glob('*/PKG-INFO'))

        # modify PKG-INFO
        with pkg_info.open() as f:
            text = f.read()
            text = text.replace(
                'Author-Email: =?utf-8?q?Jo=C3=A3o_Faria?= <joao.faria@unige.ch>',
                'Author-Email: "João Faria" <joao.faria@unige.ch>'
            )

        with pkg_info.open('wb') as f:
            f.write(text.encode())

        # replace archive, from all files in tempdir
        with tarfile.open(tar_gz, "w:gz") as w:
            for f in tdp.iterdir():
                w.add(f, arcname=f.name)

def replace_tilde_in_whl(whl):
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)

        # extract archive to temporry directory
        with zipfile.ZipFile(whl) as r:
            r.extractall(td)

        metadata = next(tdp.glob('*.dist-info/METADATA'))

        # modify METADATA
        with metadata.open() as f:
            text = f.read()
            text = text.replace(
                'Author-Email: =?utf-8?q?Jo=C3=A3o_Faria?= <joao.faria@unige.ch>',
                'Author-Email: "João Faria" <joao.faria@unige.ch>'
            )

        with metadata.open('wb') as f:
            f.write(text.encode())

        # replace archive, from all files in tempdir
        with zipfile.ZipFile(whl, "w") as w:
            for f in tdp.rglob('*'):
                w.write(f, arcname=f.relative_to(tdp))

if 'dist' in sys.argv:
    tar_gz = glob('dist/*.tar.gz')[0]
    replace_tilde_in_targz(tar_gz)

if 'wheel' in sys.argv:
    for whl in glob('wheelhouse/*.whl'):
        replace_tilde_in_whl(whl)
