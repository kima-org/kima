from numpy import array, unique

def plot_RVData(data, **kwargs):
    """ Simple plot of RV data. **kwargs are passed to plt.errorbar() """
    import matplotlib.pyplot as plt
    t = array(data.t)
    y = array(data.y)
    e = array(data.sig)
    obs = array(data.obsi)
    sb2 = data.double_lined
    if sb2:
        y2 = array(data.y2)
        e2 = array(data.sig2)


    time_offset = False
    if t[0] > 24e5:
        time_offset = True
        t -= 24e5

    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    kw = dict(fmt='o', ms=3)
    kw.update(**kwargs)

    if data.multi:
        uobs = unique(obs)
        for i in uobs:
            mask = obs == i
            ax.errorbar(t[mask], y[mask], e[mask], **kw)  
            if sb2:
                ax.errorbar(t[mask], y2[mask], e2[mask], mfc='none', **kw)  
    else:
        ax.errorbar(t, y, e,  **kw)
        if sb2:
            ax.errorbar(t, y2, e2, mfc='none', **kw)

    ax.legend(data.instruments)

    if time_offset:
        ax.set(xlabel='BJD - 2400000 [days]', ylabel='RV [m/s]')
    else:
        ax.set(xlabel='Time [days]', ylabel='RV [m/s]')
    return fig, ax



def plot_HGPMdata(data, pm_ra_bary=None, pm_dec_bary=None, 
                  show_legend=True, **kwargs):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, width_ratios=[4, 0.6, 4, 0.6], #height_ratios=[4, 2], 
                            constrained_layout=True, figsize=(7, 3))

    if pm_ra_bary is None and pm_dec_bary is None:
        f1, f2 = 1.0, 1.0
    else:
        f1, f2 = pm_ra_bary, pm_dec_bary

    kwH = dict(fmt='o', ms=4, color='C0')
    kwG = dict(fmt='o', ms=4, color='C1')
    kw = dict(fmt='o', ms=4, color='k')

    axs[0].errorbar(data.epoch_ra_hip - 5e4, data.pm_ra_hip, data.sig_hip_ra, **kwH)
    axs[0].errorbar(data.epoch_ra_gaia - 5e4, data.pm_ra_gaia, data.sig_gaia_ra, **kwG)
    axs[1].errorbar(0.5, data.pm_ra_hg, data.sig_hg_ra, **kw, mfc="w")
    axs[1].axhline(data.pm_ra_hg, color="k", zorder=-1)
    axs[1].set(yticks=[], xticks=[], xlim=(0, 1))
    axs[1].sharey(axs[0])
    axs[1].tick_params(left=False, labelleft=False)

    axs[2].errorbar(data.epoch_dec_hip - 5e4, data.pm_dec_hip, data.sig_hip_dec, **kwH)
    axs[2].errorbar(data.epoch_dec_gaia - 5e4, data.pm_dec_gaia, data.sig_gaia_dec, **kwG)
    axs[3].errorbar(0.5, data.pm_dec_hg, data.sig_hg_dec, **kw, mfc='w')
    axs[3].axhline(data.pm_dec_hg, color='k', zorder=-1)
    axs[3].set(yticks=[], xticks=[])
    axs[3].sharey(axs[2])
    axs[3].tick_params(left=False, labelleft=False)

    axs[0].set(xlabel='Epoch [BJD - 2450000]', ylabel=r'$\mu$ RA [mas/yr]')
    axs[2].set(xlabel='Epoch [BJD - 2450000]', ylabel=r'$\mu$ Dec [mas/yr]')
    # mi, ma = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0]), max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
    # axs[0].set(ylim=(mi, ma))
    # axs[1].set(ylim=(mi, ma))
    # axs[1, 1].axis('off')
    # axs[1, 3].axis('off')

    if show_legend:
        axs[0].legend(['Hipparcos', 'Gaia'], ncols=2,
                      bbox_to_anchor=(0, 1.11), loc='upper left')
    return fig, axs