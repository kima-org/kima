import os

def cli_run():
    print('running')


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