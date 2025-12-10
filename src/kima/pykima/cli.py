import os
import sys
from argparse import ArgumentParser
import tempfile

from matplotlib import pyplot as plt

import kima
from kima import RVData, RVmodel
from kima.pykima.utils import chdir

def cli_run():
    from argparse import ArgumentParser

    main_parser = ArgumentParser(add_help=False)

    group_model = main_parser.add_argument_group("model parameters")
    group_model.add_argument('--fix', help='Fix the number of Keplerians', action='store_true')
    group_model.add_argument('--npmax', help='Maximum number of Keplerians', type=int, default=1)
    group_model.add_argument('--studentt', help='Use Student-t likelihood', action='store_true')

    group_run = main_parser.add_argument_group("run parameters")
    group_run.add_argument('--steps', help='Number of steps to run', type=int, default=1000)
    group_run.add_argument('--threads', help='Number of cpu threads to use', type=int, default=8)
    group_run.add_argument('--num_pa rticles', type=int, default=1)
    group_run.add_argument('--new_level_interval', type=int, default=2000)
    group_run.add_argument('--save_interval', type=int, default=100)
    group_run.add_argument('--thread_steps', type=int, default=10)
    group_run.add_argument('--max_num_levels', type=int, default=0)
    group_run.add_argument('--lambda', type=int, default=0)
    group_run.add_argument('--beta', type=int, default=0)
    group_run.add_argument('--seed', type=int, default=0)
    group_run.add_argument('--print_thin', type=int, default=200)

    group_output = main_parser.add_argument_group("output")
    group_output.add_argument('-o', '--output', help='Name of pickle file where to save model', type=str)
    group_output.add_argument('-d', '--diagnostic', help='Show diagnostic plots', action='store_true')

    parser = ArgumentParser(parents=[main_parser])

    subparsers = parser.add_subparsers(dest='command')
    file_parser = subparsers.add_parser('file', help='Run kima on local file(s)', parents=[main_parser])
    file_parser.add_argument('-f', '--file', type=str, nargs='+')

    star_parser = subparsers.add_parser('star', help='Run kima on a star', parents=[main_parser])
    star_parser.add_argument('-s', '--star', type=str, help='Name of star for which to run kima')
    star_parser.add_argument('-i', '--instrument', type=str, dest='inst', help='Instrument to use')

    args = parser.parse_args()
    print(args)

    match args.command:
        case 'file':
            print('file')
        case 'star':
            try:
                from arvi import RV
            except ModuleNotFoundError:
                msg = 'arvi (https://github.com/j-faria/arvi) must be installed to run kima on a star'
                print(f'ModuleNotFoundError: {msg}')
                sys.exit(1)

            print(f'Querying DACE for {args.star}...')
            try:
                s = RV(args.star, instrument=args.inst,
                    do_adjust_means=False, verbose=False)
            except ValueError as e:
                print(f'Error: {e}')
                sys.exit(1)

            s.remove_instrument('HARPS', strict=True)

            print(f'--> {s}')

            if s.mtime.size == 1:
                print('Error: only one observation. Stopping.')
                sys.exit(1)

            with tempfile.TemporaryDirectory() as tmpdir:
                print(f'Writing data files and running kima in {tmpdir}')
                files = s.save(directory=tmpdir)
                files = [os.path.join(tmpdir, f) for f in files]
                data = RVData(files, skip=2)
                model = RVmodel(fix=args.fix, npmax=args.npmax, data=data)
                model.directory = tmpdir
                with chdir(model.directory):
                    kima.run(model, steps=args.steps, num_threads=args.threads)

                res = kima.load_results(model, diagnostic=args.diagnostic)
                res.save_pickle(filename=args.output)



def cli_clean(check=True, output=True):
    """
    Delete the .txt files generated during a run:
        (sample.txt, sample_info.txt, levels.txt, sampler_state.txt)

    If `output` is True, also delete output files:
        (posterior_sample.txt, posterior_sample_info.txt, weights.txt, kima_model_setup.txt)
    """
    files = ['sample.txt', 'sample_info.txt', 'levels.txt', 'sampler_state.txt']
    if output:
        files.append('posterior_sample.txt')
        files.append('posterior_sample_info.txt')
        files.append('weights.txt')
        files.append('kima_model_setup.txt')

    files = [f for f in files if os.path.exists(f)]

    if check:
        print(files)
        yn = input('Delete files? (Y/n) ')
        if yn.lower() == 'n':
            return
    for f in files:
        os.remove(f)