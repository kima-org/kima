import copy
import numpy as np
import matplotlib.pyplot as plt
from .loading import my_loadtxt, loadtxt_rows

try:
    from tqdm import trange
except ImportError:
    def trange(*args):
        return range(*args)


def logsumexp(values):
    biggest = np.max(values)
    x = values - biggest
    result = np.log(np.sum(np.exp(x))) + biggest
    return result


def logdiffexp(x1, x2):
    biggest = x1
    xx1 = x1 - biggest
    xx2 = x2 - biggest
    result = np.log(np.exp(xx1) - np.exp(xx2)) + biggest
    return result


def postprocess(temperature=1., numResampleLogX=1, plot=True, loaded=[], cut=0,
                save=True, zoom_in=True, compression_bias_min=1.0,
                verbose=True, compression_scatter=0.0, moreSamples=1,
                compression_assert=None, single_precision=False):

    if len(loaded) == 0:
        levels_orig = np.atleast_2d(my_loadtxt("levels.txt"))
        sample_info = np.atleast_2d(my_loadtxt("sample_info.txt"))
    else:
        levels_orig, sample_info = loaded[0], loaded[1]

    # Remove regularisation from levels_orig if we asked for it
    if compression_assert is not None:
        ones = np.ones(levels_orig.shape[0] - 1)
        levels_orig[1:, 0] = -np.cumsum(compression_assert * ones)

    cut = int(cut * sample_info.shape[0])
    sample_info = sample_info[cut:, :]

    if plot:
        _, ax = plt.subplots(1, 1)
        ax.plot(sample_info[:, 0], "k")
        ax.set(xlabel="Iteration", ylabel="Level",
               title='DNest4: level of each saved particle')

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(np.diff(levels_orig[:, 0]), "k")
        ax1.set(ylabel="Compression", xlabel="Level",
                title='DNest4: compression factor between levels')
        xlim = ax1.get_xlim()
        ax1.axhline(-1., color='g')
        ax1.axhline(-np.log(10.), color='g', linestyle="--")
        ax1.set_ylim(ymax=0.05)

        good = np.nonzero(levels_orig[:, 4] > 0)[0]
        ax2.plot(levels_orig[good, 3] / levels_orig[good, 4], "ko-")
        ax2.set(xlim=xlim, ylim=[0, 1], xlabel="Level", ylabel="MH Acceptance",
                title='DNest4: MCMC acceptance fraction for each level')
        fig.tight_layout()

    # Convert to lists of tuples
    logl_levels = [(levels_orig[i, 1], levels_orig[i, 2])
                   for i in range(0, levels_orig.shape[0])]  # logl, tiebreaker
    logl_samples = [(sample_info[i, 1], sample_info[i, 2], i)
                    for i in range(0, sample_info.shape[0])
                    ]  # logl, tiebreaker, id
    logx_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logp_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logP_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    P_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logz_estimates = np.zeros((numResampleLogX, 1))
    H_estimates = np.zeros((numResampleLogX, 1))

    # Find sandwiching level for each sample
    sandwich = sample_info[:, 0].copy().astype('int')
    for i in range(0, sample_info.shape[0]):
        while (sandwich[i] < levels_orig.shape[0] - 1) and (
                logl_samples[i] > logl_levels[sandwich[i] + 1]):
            sandwich[i] += 1

    for z in range(0, numResampleLogX):
        # Make a monte carlo perturbation of the level compressions
        levels = levels_orig.copy()
        compressions = -np.diff(levels[:, 0])
        compressions *= compression_bias_min + (
            1.0 - compression_bias_min) * np.random.rand()
        compressions *= np.exp(compression_scatter *
                               np.random.randn(compressions.size))
        levels[1:, 0] = -compressions
        levels[:, 0] = np.cumsum(levels[:, 0])

        # For each level
        for i in range(0, levels.shape[0]):
            # Find the samples sandwiched by this level
            which = np.nonzero(sandwich == i)[0]
            logl_samples_thisLevel = []  # (logl, tieBreaker, ID)
            for j in range(0, len(which)):
                logl_samples_thisLevel.append(
                    copy.deepcopy(logl_samples[which[j]]))
            logl_samples_thisLevel = sorted(logl_samples_thisLevel)
            N = len(logl_samples_thisLevel)

            # Generate intermediate logx values
            logx_max = levels[i, 0]
            if i == levels.shape[0] - 1:
                logx_min = -1E300
            else:
                logx_min = levels[i + 1, 0]
            Umin = np.exp(logx_min - logx_max)

            if N == 0 or numResampleLogX > 1:
                U = Umin + (1 - Umin) * np.random.rand(len(which))
            else:
                U = Umin + (1 - Umin) * np.linspace(1 / (N + 1), 1 - 1 /
                                                    (N + 1), N)
            logx_samples_thisLevel = np.sort(logx_max + np.log(U))[::-1]

            for j in range(0, which.size):
                logx_samples[logl_samples_thisLevel[j]
                             [2]][z] = logx_samples_thisLevel[j]

                if j != which.size - 1:
                    left = logx_samples_thisLevel[j + 1]
                elif i == levels.shape[0] - 1:
                    left = -1E300
                else:
                    left = levels[i + 1][0]

                if j != 0:
                    right = logx_samples_thisLevel[j - 1]
                else:
                    right = levels[i][0]

                logp_samples[logl_samples_thisLevel[j]
                             [2]][z] = np.log(0.5) + logdiffexp(right, left)

        logl = sample_info[:, 1] / temperature

        logp_samples[:, z] = logp_samples[:, z] - logsumexp(logp_samples[:, z])
        logP_samples[:, z] = logp_samples[:, z] + logl
        logz_estimates[z] = logsumexp(logP_samples[:, z])
        logP_samples[:, z] -= logz_estimates[z]
        P_samples[:, z] = np.exp(logP_samples[:, z])
        H_estimates[z] = -logz_estimates[z] + np.sum(P_samples[:, z] * logl)

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(logx_samples[:, z], sample_info[:, 1], 'k.',
                     label='Samples')
            ax1.plot(levels[1:, 0], levels[1:, 1], 'g.', label='Levels')
            ax1.legend(numpoints=1, loc='lower left')
            title = ('DNest4: Log-likelihood vs enclosed prior mass '
                     'for each sample/level')
            ax1.set(ylabel='log(L)', title=title)

            # fig.suptitle(str(z+1) + "/" + str(numResampleLogX) + ", log(Z) = " + str(logz_estimates[z][0]))
            fig.suptitle("log(Z) = %7.3f" % logz_estimates[z][0])

            # Use all plotted logl values to set ylim
            combined_logl = np.hstack([sample_info[:, 1], levels[1:, 1]])
            combined_logl = np.sort(combined_logl)
            lower = combined_logl[int(0.1 * combined_logl.size)]
            upper = combined_logl[-1]
            diff = upper - lower
            lower -= 0.05 * diff
            upper += 0.05 * diff
            if zoom_in:
                ax1.set_ylim([lower, upper])
            xlim = ax1.get_xlim()

            ax2.plot(logx_samples[:, z], P_samples[:, z], 'k.')
            ax2.set(ylabel='Posterior Weights', xlabel='log(X)', xlim=xlim,
                    title='DNest4: Posterior weight of each sample')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    P_samples = np.mean(P_samples, 1)
    P_samples = P_samples / np.sum(P_samples)
    logz_estimate = np.mean(logz_estimates)
    logz_error = np.std(logz_estimates)
    H_estimate = np.mean(H_estimates)
    H_error = np.std(H_estimates)
    ESS = np.exp(-np.sum(P_samples * np.log(P_samples + 1E-300)))

    errorbar1 = ""
    errorbar2 = ""
    if numResampleLogX > 1:
        errorbar1 += " +- " + str(logz_error)
        errorbar2 += " +- " + str(H_error)

    if verbose:
        print("log(Z) = " + str(logz_estimate) + errorbar1)
        print("Information = " + str(H_estimate) + errorbar2 + " nats.")
        print("Effective sample size = " + str(ESS))

    # print(f'up to here: {time() - start} sec')

    # Resample to uniform weight
    N = int(moreSamples * ESS)
    w = P_samples
    w = w / np.max(w)
    rows = np.empty(N, dtype="int64")
    for i in trange(0, N):
        while True:
            which = np.random.randint(sample_info.shape[0])
            if np.random.rand() <= w[which]:
                break
        rows[i] = which + cut

    # Get header rows
    f1 = open("sample.txt", "r")
    line = f1.readline()
    if line[0] == "#":
        header = line[1:]
    else:
        header = ""
    f1.close()
    f2 = open("sample_info.txt", "r")
    line = f2.readline()
    if line[0] == "#":
        header_info = line[1:]
    else:
        header_info = ""
    f2.close()

    sample = loadtxt_rows("sample.txt", set(rows), single_precision)
    sample_info = loadtxt_rows("sample_info.txt", set(rows), single_precision)
    posterior_sample = None
    posterior_sample_lnlike = None
    if single_precision:
        posterior_sample = np.empty((N, sample["ncol"]), dtype="float32")
        posterior_sample_lnlike = np.empty((N, sample_info["ncol"]),
                                           dtype="float32")
    else:
        posterior_sample = np.empty((N, sample["ncol"]))
        posterior_sample_lnlike = np.empty((N, sample_info["ncol"]))

    for i in range(0, N):
        posterior_sample[i, :] = sample[rows[i]]
        posterior_sample_lnlike[i, :] = sample_info[rows[i]]

    if save:
        np.savetxt('weights.txt', w)
        if single_precision:
            np.savetxt("posterior_sample.txt", posterior_sample, fmt="%.7e",
                       header=header)
            np.savetxt("posterior_sample_info.txt", posterior_sample_lnlike,
                       fmt="%.7e", header=header_info)
        else:
            np.savetxt("posterior_sample.txt", posterior_sample, header=header)
            np.savetxt("posterior_sample_info.txt", posterior_sample_lnlike,
                       fmt=['%d', '%f', '%f', '%d'], header=header_info)

    if plot:
        plt.show()

    # print(f'up to here: {time() - start} sec')
    return [logz_estimate, H_estimate, logx_samples]


def postprocess_abc(temperature=1., numResampleLogX=1, plot=True, loaded=[],
                    cut=0., save=True, zoom_in=True, compression_bias_min=1.,
                    verbose=True, compression_scatter=0., moreSamples=1.,
                    compression_assert=None, single_precision=False,
                    threshold_fraction=0.8):
    if len(loaded) == 0:
        levels_orig = np.atleast_2d(my_loadtxt("levels.txt"))
        sample_info = np.atleast_2d(my_loadtxt("sample_info.txt"))
    else:
        levels_orig, sample_info = loaded[0], loaded[1]

    # Remove regularisation from levels_orig if we asked for it
    if compression_assert is not None:
        levels_orig[1:, 0] = -np.cumsum(
            compression_assert * np.ones(levels_orig.shape[0] - 1))

    cut = int(cut * sample_info.shape[0])
    sample_info = sample_info[cut:, :]

    if plot:
        plt.figure(1)
        plt.plot(sample_info[:, 0], "k")
        plt.xlabel("Iteration")
        plt.ylabel("Level")

        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(np.diff(levels_orig[:, 0]), "k")
        plt.ylabel("Compression")
        plt.xlabel("Level")
        xlim = plt.gca().get_xlim()
        plt.axhline(-1., color='g')
        plt.axhline(-np.log(10.), color='g', linestyle="--")
        plt.ylim(ymax=0.05)

        plt.subplot(2, 1, 2)
        good = np.nonzero(levels_orig[:, 4] > 0)[0]
        plt.plot(levels_orig[good, 3] / levels_orig[good, 4], "ko-")
        plt.xlim(xlim)
        plt.ylim([0., 1.])
        plt.xlabel("Level")
        plt.ylabel("MH Acceptance")

    # Convert to lists of tuples
    logl_levels = [(levels_orig[i, 1], levels_orig[i, 2])
                   for i in range(0, levels_orig.shape[0])
                   ]  # logl, tiebreakercut
    logl_samples = [(sample_info[i, 1], sample_info[i, 2], i)
                    for i in range(0, sample_info.shape[0])
                    ]  # logl, tiebreaker, id
    logx_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logp_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logP_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    P_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logz_estimates = np.zeros((numResampleLogX, 1))
    H_estimates = np.zeros((numResampleLogX, 1))

    # Find sandwiching level for each sample
    sandwich = sample_info[:, 0].copy().astype('int')
    for i in range(0, sample_info.shape[0]):
        while sandwich[i] < levels_orig.shape[0] - 1 and logl_samples[
                i] > logl_levels[sandwich[i] + 1]:
            sandwich[i] += 1

    for z in range(0, numResampleLogX):
        # Make a monte carlo perturbation of the level compressions
        levels = levels_orig.copy()
        compressions = -np.diff(levels[:, 0])
        compressions *= compression_bias_min + (
            1. - compression_bias_min) * np.random.rand()
        compressions *= np.exp(compression_scatter *
                               np.random.randn(compressions.size))
        levels[1:, 0] = -compressions
        levels[:, 0] = np.cumsum(levels[:, 0])

        # For each level
        for i in range(0, levels.shape[0]):
            # Find the samples sandwiched by this level
            which = np.nonzero(sandwich == i)[0]
            logl_samples_thisLevel = []  # (logl, tieBreaker, ID)
            for j in range(0, len(which)):
                logl_samples_thisLevel.append(
                    copy.deepcopy(logl_samples[which[j]]))
            logl_samples_thisLevel = sorted(logl_samples_thisLevel)
            N = len(logl_samples_thisLevel)

            # Generate intermediate logx values
            logx_max = levels[i, 0]
            if i == levels.shape[0] - 1:
                logx_min = -1E300
            else:
                logx_min = levels[i + 1, 0]
            Umin = np.exp(logx_min - logx_max)

            if N == 0 or numResampleLogX > 1:
                U = Umin + (1. - Umin) * np.random.rand(len(which))
            else:
                U = Umin + (1. - Umin) * np.linspace(1. / (N + 1), 1. - 1. /
                                                     (N + 1), N)
            logx_samples_thisLevel = np.sort(logx_max + np.log(U))[::-1]

            for j in range(0, which.size):
                logx_samples[logl_samples_thisLevel[j]
                             [2]][z] = logx_samples_thisLevel[j]

                if j != which.size - 1:
                    left = logx_samples_thisLevel[j + 1]
                elif i == levels.shape[0] - 1:
                    left = -1E300
                else:
                    left = levels[i + 1][0]

                if j != 0:
                    right = logx_samples_thisLevel[j - 1]
                else:
                    right = levels[i][0]

                logp_samples[logl_samples_thisLevel[j]
                             [2]][z] = np.log(0.5) + logdiffexp(right, left)

        logl = sample_info[:, 1] / temperature

        logp_samples[:, z] = logp_samples[:, z] - logsumexp(logp_samples[:, z])

        # Define the threshold for ABC, in terms of log(X)
        threshold = threshold_fraction * levels[:, 0].min()

        # Particles below threshold get no posterior weight
        logp_samples[logx_samples > threshold] = -1E300

        logP_samples[:, z] = logp_samples[:, z] + logl
        logz_estimates[z] = logsumexp(logP_samples[:, z])
        logP_samples[:, z] -= logz_estimates[z]
        P_samples[:, z] = np.exp(logP_samples[:, z])
        H_estimates[z] = -logz_estimates[z] + np.sum(P_samples[:, z] * logl)

        if plot:
            plt.figure(3)

            plt.subplot(2, 1, 1)
            plt.plot(logx_samples[:, z], sample_info[:, 1], 'k.',
                     label='Samples')
            plt.plot(levels[1:, 0], levels[1:, 1], 'g.', label='Levels')
            plt.legend(numpoints=1, loc='lower left')
            plt.ylabel('log(L)')
            plt.title(
                str(z + 1) + "/" + str(numResampleLogX) + ", log(Z) = " +
                str(logz_estimates[z][0]))
            # Use all plotted logl values to set ylim
            combined_logl = np.hstack([sample_info[:, 1], levels[1:, 1]])
            combined_logl = np.sort(combined_logl)
            lower = combined_logl[int(0.1 * combined_logl.size)]
            upper = combined_logl[-1]
            diff = upper - lower
            lower -= 0.05 * diff
            upper += 0.05 * diff
            if zoom_in:
                plt.ylim([lower, upper])

            xlim = plt.gca().get_xlim()

        if plot:
            plt.subplot(2, 1, 2)
            plt.plot(logx_samples[:, z], P_samples[:, z], 'k.')
            plt.ylabel('Posterior Weights')
            plt.xlabel('log(X)')
            plt.xlim(xlim)

    P_samples = np.mean(P_samples, 1)
    P_samples = P_samples / np.sum(P_samples)
    logz_estimate = np.mean(logz_estimates)
    logz_error = np.std(logz_estimates)
    H_estimate = np.mean(H_estimates)
    H_error = np.std(H_estimates)
    ESS = np.exp(-np.sum(P_samples * np.log(P_samples + 1E-300)))

    errorbar1 = ""
    errorbar2 = ""
    if numResampleLogX > 1:
        errorbar1 += " +- " + str(logz_error)
        errorbar2 += " +- " + str(H_error)

    if verbose:
        print("log(Z) = " + str(logz_estimate) + errorbar1)
        print("Information = " + str(H_estimate) + errorbar2 + " nats.")
        print("Effective sample size = " + str(ESS))

    # Resample to uniform weight
    N = int(moreSamples * ESS)
    w = P_samples
    w = w / np.max(w)
    rows = np.empty(N, dtype="int64")
    for i in range(0, N):
        while True:
            which = np.random.randint(sample_info.shape[0])
            if np.random.rand() <= w[which]:
                break
        rows[i] = which + cut

    sample = loadtxt_rows("sample.txt", set(rows), single_precision)
    posterior_sample = None
    if single_precision:
        posterior_sample = np.empty((N, sample["ncol"]), dtype="float32")
    else:
        posterior_sample = np.empty((N, sample["ncol"]))

    for i in range(0, N):
        posterior_sample[i, :] = sample[rows[i]]

    if save:
        np.savetxt('weights.txt', w)
        if single_precision:
            np.savetxt("posterior_sample.txt", posterior_sample, fmt="%.7e")
        else:
            np.savetxt("posterior_sample.txt", posterior_sample)

    if plot:
        plt.show()

    return [logz_estimate, H_estimate, logx_samples]


def diffusion_plot():
    """
    Plot a nice per-particle diffusion plot.
    """

    sample_info = np.atleast_2d(my_loadtxt('sample_info.txt'))
    ID = sample_info[:, 3].astype('int')
    j = sample_info[:, 0].astype('int')

    ii = np.arange(1, sample_info.shape[0] + 1)

    for i in range(0, ID.max() + 1):
        which = np.nonzero(ID == i)[0]
        plt.plot(ii[which], j[which])

    plt.xlabel('Iteration')
    plt.ylabel('Level')
    plt.show()


def levels_plot():
    """
    Plot the differences between the logl values of the levels.
    """
    levels = my_loadtxt('levels.txt')

    plt.plot(np.log10(np.diff(levels[:, 1])), "ko-")
    plt.ylim([-1, 4])
    plt.axhline(0., color='g', linewidth=2)
    plt.axhline(np.log10(np.log(10.)), color='g')
    plt.axhline(np.log10(0.8), color='g', linestyle='--')
    plt.xlabel('Level')
    plt.ylabel('$\\log_{10}$(Delta log likelihood)')
    plt.show()
