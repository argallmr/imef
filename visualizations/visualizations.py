import numpy as np
import xarray as xr
from matplotlib import pyplot as plt, dates as mdates, ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

R_E = 6371.2  # Earth radius (km)


def add_colorbar(ax, im, wpad=1.05):
    '''
    Add a colorbar to the axes.
    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes to which the colorbar is attached.
    im : `matplotlib.axes.Axes.pcolorfast`
        The image that the colorbar will represent.
    '''
    cbaxes = inset_axes(ax,
                        width='2%', height='100%', loc=4,
                        bbox_to_anchor=(0, 0, wpad, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
    cb = plt.colorbar(im, cax=cbaxes, orientation='vertical')
    cb.ax.minorticks_on()

    return cb


def add_legend(ax, lines=None, corner='NE', outside=False, horizontal=False, title=None):
    '''
    Add a legend to the axes. Legend elements will have the same color as the
    lines that they label.
    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes to which the legend is attached.
    lines : list of `matplotlib.lines.Line2D`
        The line elements that the legend format should match
    corner : str
        The bounding box of the legend will be tied to this corner:
        ('NE', 'NW', 'SE', 'SW')
    outside : bool
        The bounding box will extend outside the plot
    horizontal : bool
        The legend items will be placed in columns (side-by-side) instead of
        rows (stacked vertically)
    '''

    if horizontal:
        ncol = len(lines)
        columnspacing = 0.5
    else:
        ncol = 1
        columnspacing = 0.0

    if corner == 'NE':
        bbox_to_anchor = (1, 1)
        loc = 'upper left' if outside else 'upper right'
    elif corner == 'SE':
        bbox_to_anchor = (1, 0)
        loc = 'lower left' if outside else 'lower right'
    elif corner == 'NW':
        bbox_to_anchor = (0, 1)
        loc = 'upper right' if outside else 'upper left'
    elif corner == 'SW':
        bbox_to_anchor = (0, 0)
        loc = 'lower right' if outside else 'lower left'

    leg = ax.legend(bbox_to_anchor=bbox_to_anchor,
                    borderaxespad=0.0,
                    columnspacing=columnspacing,
                    frameon=False,
                    handlelength=1,
                    handletextpad=0.25,
                    loc=loc,
                    ncol=ncol,
                    title=title)


#    for line, text in zip(lines, leg.get_texts()):
#        text.set_color(line.get_color())


def draw_earth_cart(ax):
    '''
    A handy function for drawing the Earth in a set of cartesian axes
    '''
    N = 30
    r = np.ones(N)

    # Closed semi-circile on night-side
    theta = np.linspace(-np.pi / 2, np.pi / 2, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.fill_betweenx(y, x, np.zeros(N), color='k')

    #  Open semi-circle on dayside
    theta = np.linspace(np.pi / 2, 3 * np.pi / 2, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, color='k')


def draw_earth_pol(ax):
    '''
    A handy function for drawing the Earth in a set of Polar Axes
    '''
    ax.fill_between(np.linspace(-np.pi / 2, np.pi / 2, 30), 0, np.ones(30), color='k')
    ax.plot(np.linspace(np.pi / 2, 3 * np.pi / 2, 30), np.ones(30), color='k')


def format_axes(ax, xaxis=True, yaxis=True, time=True):
    '''
    Format the abcissa and ordinate axes
    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Axes to be formatted
    time : bool
        If true, format the x-axis with dates
    xaxis, yaxis : str
        Indicate how the axes should be formatted. Options are:
        ('on', 'time', 'off'). If 'time', the ConciseDateFormatter is applied
        If 'off', the axis label and ticklabels are suppressed. If 'on', the
        default settings are used
    '''
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    # All x-axes should be formatted with time
    if time:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    else:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    if not xaxis:
        ax.set_xticklabels([])
        ax.set_xlabel('')

    if ax.get_yscale() == 'log':
        locmaj = ticker.LogLocator(base=10.0)
        ax.yaxis.set_major_locator(locmaj)

        locmin = ticker.LogLocator(base=10.0, subs=(0.3, 0.6, 0.9))
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    if not yaxis:
        ax.set_yticklabels([])
        ax.set_ylabel('')


def plot_all_L(ds):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6.5, 7.5))
    plt.subplots_adjust(top=0.95, bottom=0.1, right=0.85, hspace=0.2, wspace=0.4)

    # EDI
    ax = axes[0, 0]
    plot_L(ds['L'].reindex_like(ds['E_EDI']), ds['E_EDI'],
           ax=ax, L_sc=ds['L'], title='EDI', cblabel='E (mV/m)')
    format_axes(ax, xaxis=False)

    # EDP
    ax = axes[0, 1]
    plot_L(ds['L'].reindex_like(ds['E_EDP']), ds['E_EDP'],
           ax=ax, L_sc=ds['L'], title='EDP', cblabel='E (mV/m)')
    format_axes(ax, xaxis=False, yaxis=False)

    # DIS
    ax = axes[1, 0]
    plot_L(ds['L'].reindex_like(ds['E_DIS']), ds['E_DIS'],
           ax=ax, L_sc=ds['L'], title='DIS', cblabel='E (mV/m)')
    format_axes(ax, xaxis=False)

    # DES
    ax = axes[1, 1]
    plot_L(ds['L'].reindex_like(ds['E_DES']), ds['E_DES'],
           ax=ax, L_sc=ds['L'], title='DES', cblabel='E (mV/m)')
    format_axes(ax, xaxis=False, yaxis=False)

    # Convection
    ax = axes[2, 0]
    plot_L(ds['L'].reindex_like(ds['E_con']), ds['E_con'],
           ax=ax, L_sc=ds['L'], title='Spacecraft', cblabel='E (mV/m)')
    format_axes(ax)

    # Corotation
    ax = axes[2, 1]
    plot_L(ds['L'].reindex_like(ds['E_cor']), ds['E_cor'],
           ax=ax, L_sc=ds['L'], title='Corotation', cblabel='E (mV/m)')
    format_axes(ax, yaxis=False)

    return fig, axes


def plot_all_orbit_quiver(ds):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(5.5, 7.5))
    fig.suptitle('E-field (mV/m) along Orbit Track')
    plt.subplots_adjust(top=0.92, bottom=0.08, right=0.98, hspace=0.2, wspace=0.02)

    # EDI
    ax = axes[0, 0]
    plot_orbit_quiver(ds['R_sc'], ds['R_sc'].reindex_like(ds['E_EDI']), ds['E_EDI'],
                      ax=ax, title='EDI')
    format_axes(ax, xaxis=False, time=False)

    # EDP
    ax = axes[0, 1]
    plot_orbit_quiver(ds['R_sc'], ds['R_sc'].reindex_like(ds['E_EDP']), ds['E_EDP'],
                      ax=ax, title='EDP')
    format_axes(ax, xaxis=False, yaxis=False, time=False)

    # DIS
    ax = axes[1, 0]
    plot_orbit_quiver(ds['R_sc'], ds['R_sc'].reindex_like(ds['E_DIS']), ds['E_DIS'],
                      ax=ax, title='DIS')
    format_axes(ax, xaxis=False, time=False)

    # DES
    ax = axes[1, 1]
    plot_orbit_quiver(ds['R_sc'], ds['R_sc'].reindex_like(ds['E_DES']), ds['E_DES'],
                      ax=ax, title='DES')
    format_axes(ax, xaxis=False, yaxis=False, time=False)

    # Convection
    ax = axes[2, 0]
    plot_orbit_quiver(ds['R_sc'], ds['R_sc'].reindex_like(ds['E_con']), ds['E_con'],
                      ax=ax, title='Spacecraft')
    format_axes(ax, time=False)

    # Corotation
    ax = axes[2, 1]
    plot_orbit_quiver(ds['R_sc'], ds['R_sc'].reindex_like(ds['E_cor']), ds['E_cor'],
                      ax=ax, title='Corotation')
    format_axes(ax, yaxis=False, time=False)

    return fig, axes


def plot_cross_corr(vec1, vec2, label1='', label2='', title=''):
    '''
    Create a figure comparing the components of two different vectors.

    Parameters
    ----------
    vec1, vec2 : `xarray.DataArray`
        A time series vector field of data
    label1, label2 : str
        Label for the data
    title : str
        Plot title

    Returns
    -------
    fig : figure object
        Figure
    axes : list of subplots
        Subplot axes
    '''
    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=False)
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.8, wspace=0.6)

    axlim = max(vec1.max(), vec2.max())

    ax = axes[0, 0]
    ax.scatter(vec1[:, 0], vec2[:, 0])
    ax.plot([-axlim, axlim], [-axlim, axlim], linestyle='dashed', color='orange')
    ax.set_title('Ex (mV/m)')
    ax.set_xlabel(label1)
    ax.set_xlim(-axlim, axlim)
    ax.set_ylabel(label2)
    ax.set_ylim(-axlim, axlim)
    format_axes(ax, time=False)

    ax = axes[0, 1]
    ax.scatter(vec1[:, 1], vec2[:, 1])
    ax.plot([-axlim, axlim], [-axlim, axlim], linestyle='dashed', color='orange')
    ax.set_title('Ey (mV/m)')
    ax.set_xlim(-axlim, axlim)
    ax.set_xlabel(label1)
    ax.set_ylim(-axlim, axlim)
    format_axes(ax, time=False)

    ax = axes[0, 2]
    ax.scatter(vec1[:, 2], vec2[:, 2])
    ax.plot([-axlim, axlim], [-axlim, axlim], linestyle='dashed', color='orange')
    ax.set_title('Ez (mV/m)')
    ax.set_xlabel(label1)
    ax.set_xlim(-axlim, axlim)
    ax.set_ylim(-axlim, axlim)
    format_axes(ax, time=False)

    return fig, axes


def plot_efield_r_kp(ds, varname, MLT_range=None):
    '''
    '''
    # Subset in MLT
    if MLT_range is not None:
        theta_min = MLT_range[0] * 2 * np.pi / 24
        theta_max = MLT_range[1] * 2 * np.pi / 24
        if theta_min > theta_max:
            ds = ds.isel({'theta': (ds['theta'] >= theta_min)
                                   | (ds['theta'] <= theta_max)})
        else:
            ds = ds.isel({'theta': (ds['theta'] <= theta_min)
                                   & (ds['theta'] >= theta_max)})
    else:
        MLT_range = [0, 24]

    # Create the figure
    fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(6.5, 5))
    fig.suptitle('MLT = [{0:3.1f}, {1:3.1f}]'
                 .format(MLT_range[0], MLT_range[1]))
    plt.subplots_adjust(wspace=0.33, hspace=0.2, right=0.8, top=0.93)

    # Plot a component
    for c in ds.comp.data:
        if c == 'x':
            ax = axes[0, 0]
            ax_one = ax
            plot_counts = False
            legend = False
            label = 'Ex'
        elif c == 'y':
            ax = axes[0, 1]
            ax_one = ax
            plot_counts = False
            legend = True
            label = 'Ey'
        elif c == 'z':
            ax = axes[1, 0]
            ax_cts = axes[1, 1]
            ax_one = [ax, ax_cts]
            plot_counts = True
            legend = False
            label = 'Ez'

        plot_efield_r_kp_one(ds[varname + '_mean'].loc[..., c],
                             ds[varname + '_counts'].loc[..., c],
                             axes=ax_one, plot_counts=plot_counts, legend=legend)

        ax.set_ylabel(label + ' (mV/m)')

        if not ax.is_last_row():
            ax.set_xlabel('')
            ax.set_xticklabels([])

    return fig, axes


def plot_efield_r_kp_one(E, counts, axes=None, plot_counts=False, legend=False):
    # Create the axes
    if axes is None:
        # Number of columns
        #   - Irrelevant if axes is given
        ncols = 1
        if plot_counts:
            ncols = 2

        # Create the figure
        fig, axes = plt.subplots(nrows=1, ncols=ncols, squeeze=False, figsize=(6.5, 4))
        plt.subplots_adjust(wspace=0.33, bottom=0.15, right=0.85)

        # Assign the axes
        ax = axes[0, 0]
        if plot_counts:
            ax_cts = axes[0, 1]

    # Axes were given
    else:

        # Assign the axes
        if plot_counts:
            ax = axes[0]
            ax_cts = axes[1]
        else:
            ax = axes

        # Get the figure
        fig = ax.figure

    # Create the line plot
    r_min = np.inf
    for ikp, E_kp in enumerate(E):
        # Average over the remaining theta bins
        data = np.ma.average(E_kp.to_masked_array(), axis=1,
                             weights=counts[ikp, ...])

        # Sum all counts over the theta dimension
        cts = np.sum(counts[ikp, ...], axis=1)
        cts = cts.where(cts != 0)

        # Minimum radial distance
        try:
            r_min_temp = next(r for r, e in zip(E['r'].data, np.ma.getmask(data)) if not e)
            r_min = min(r_min, r_min_temp)
        except:
            # If there is no data in this kp range, skip this line and go to the next.
            continue

        # Label for the Kp bin
        kp_min = np.ceil(E['kp'].data[ikp])
        kp_max = np.ceil(E['kp'].data[ikp] + 1)
        #        if ikp == len(ds['kp']) - 2:
        #            kp_max = np.floor(ds['kp'][ikp+1])
        label = '{0} to {1}-'.format(kp_min, kp_max)

        # Plot E(r) at each Kp bin
        ax.plot(E_kp['r'], data, label=label, marker='*')

        # Plot counts(r) at each Kp bin
        if plot_counts:
            ax_cts.plot(E['r'], cts, label=label, marker='*')

    # Adjust the plot
    ax.set_xlabel('R ($R_{E}$)')
    ax.set_xlim(np.floor(r_min), np.ceil(E['r'][-1]))
    ax.set_ylabel('E (mV/m)')

    # Adjust the counts plot
    if plot_counts:
        ax_cts.set_xlabel('R ($R_{E}$)')
        ax_cts.set_xlim(np.floor(r_min), np.ceil(E['r'][-1]))
        ax_cts.set_ylabel('Counts')
        ax_cts.set_yscale('log')

    # Add a legend
    if legend:
        if plot_counts:
            add_legend(ax_cts, outside=True, title='Kp Index')
        else:
            add_legend(ax, outside=True, title='Kp Index')

    return fig, axes


def plot_global_counts_kp(ds, varname='E_EDI_corot'):
    fig, axes = plt.subplots(nrows=3, ncols=3, squeeze=False, figsize=(7, 7),
                             subplot_kw=dict(projection='polar'))
    plt.subplots_adjust(left=0.05, wspace=0.65, hspace=0.65, bottom=0.1)

    for idx, cts in enumerate(ds[varname + '_counts']):
        irow = idx // 3
        icol = idx % 3
        ax = axes[irow, icol]

        ax_title = 'kp=[{0},{1}-]'.format(np.ceil(ds['kp'][idx].data),
                                          np.ceil(ds['kp'][idx].data) + 1)

        plot_global_counts_one(cts[..., 0], axes=ax)

        ax.set_title(ax_title)

        if ~ax.is_last_row():
            ax.set_xlabel('')
        #            ax.set_xticklabels([])

        if ~ax.is_first_col():
            ax.set_ylabel('')

    # Last plot
    ax = axes[-1, -1]
    ax.set_title('Kp=[0,9-]')
    ax.set_xlabel('')

    # Sum all counts over the theta dimension
    cts = np.sum(ds[varname + '_counts'], axis=0)
    cts = cts.where(cts != 0)

    plot_global_counts_one(cts[..., 0], axes=ax)

    return fig, axes


def plot_global_counts_one(counts, axes=None):
    if axes is None:
        fix, axes = plt.subplots(nrows=1, ncols=1, squeeze=False,
                                 subplot_kw=dict(projection='polar'))
        ax = axes[0, 0]
    else:
        ax = axes
        fig = ax.figure

    log_counts = np.ma.log(counts)

    im = ax.pcolormesh(counts['theta'], counts['r'], log_counts,
                       cmap='YlOrRd', shading='auto')
    ax.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax.set_xlabel("log$_{10}$(Counts)")

    draw_earth_pol(ax)

    add_colorbar(ax, im, wpad=1.3)

    return fig, axes


def plot_global_efield_kp(ds, varname='E_EDI_corot'):
    '''
    Create a polar plot of the Electric field vectors binned in (r, theta) space.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Data to be plotted, including the r and theta bins.
    varname : str
        Name of the variable to be plotted

    Returns
    -------
    fig : matplotlib figure
        Figure object
    axes : matplotlib subplots
        Axes in which the counts and electric field are plotted
    '''

    fig, axes = plt.subplots(nrows=3, ncols=3, squeeze=False, figsize=(7, 7),
                             subplot_kw=dict(projection='polar'))
    plt.subplots_adjust(wspace=0.6, hspace=0.65, bottom=0.1)

    for idx, E in enumerate(ds[varname + '_mean']):
        irow = idx // 3
        icol = idx % 3
        ax = axes[irow, icol]

        ax_title = 'kp=[{0},{1}-]'.format(np.ceil(ds['kp'][idx].data),
                                          np.ceil(ds['kp'][idx].data) + 1)

        plot_global_efield_one(E, ds[varname + '_counts'][idx, ...],
                               axes=ax, plot_counts=False)

        ax.set_title(ax_title)

        if ~ax.is_last_row():
            ax.set_xlabel('')
        #            ax.set_xticklabels([])

        if ~ax.is_first_col():
            ax.set_ylabel('')

    # Last plot
    ax = axes[-1, -1]
    ax.set_title('Kp=[0,9-]')
    ax.set_xlabel('')

    # Average over the remaining theta bins
    data = xr.DataArray(np.ma.average(ds[varname + '_mean'].to_masked_array(), axis=0,
                                      weights=ds[varname + '_counts']),
                        dims=('r', 'theta', 'comp'),
                        coords={'r': ds['r'],
                                'theta': ds['theta'],
                                'comp': ds['comp']})

    # Sum all counts over the theta dimension
    cts = np.sum(ds[varname + '_counts'], axis=0)
    cts = cts.where(cts != 0)

    plot_global_efield_one(data, cts,
                           axes=ax, plot_counts=False)

    return fig, axes


def plot_global_efield_one(E, counts, axes=None, plot_counts=False):
    '''
    Create a polar plot of the Electric field vectors binned in (r, theta) space.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Data to be plotted, including the r and theta bins.
    varname : str
        Name of the variable to be plotted

    Returns
    -------
    fig : matplotlib figure
        Figure object
    axes : matplotlib subplots
        Axes in which the counts and electric field are plotted
    '''
    # Create the axes
    if axes is None:
        # Number of columns
        #   - Irrelevant if axes is given
        ncols = 1
        if plot_counts:
            ncols = 2

        # Create the figure
        fig, axes = plt.subplots(nrows=1, ncols=ncols, squeeze=False, figsize=(6.5, 4),
                                 subplot_kw=dict(projection='polar'))
        plt.subplots_adjust(wspace=0.33, bottom=0.15, right=0.85)

        # Assign the axes
        ax = axes[0, 0]
        if plot_counts:
            ax_cts = axes[0, 1]

    # Axes were given
    else:

        # Assign the axes
        if plot_counts:
            ax = axes[0]
            ax_cts = axes[1]
        else:
            ax = axes

        # Get the figure
        fig = ax.figure

    # Global Electric Field
    ax.quiver(E['theta'], E['r'], E['E_convective_mean'][..., 0], E['E_convective_mean'][..., 1])
    # ax.quiver(E['theta'], E['r'], E[..., 0], E[..., 1])
    ax.set_xlabel("Electric Field")
    ax.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax.set_theta_direction(1)

    # Draw the earth
    draw_earth_pol(ax)

    # Counts
    if plot_counts:
        plot_global_counts_one(counts[..., 0], axes=ax_cts)

    return fig, axes


def plot_L(L, E, ax=None, L_sc=None, title='', cblabel=''):
    '''
    Plot the electric field magnitude as a function of L

    Parameters
    ----------
    L : `xarray.DataArray`
        Geocentric distance to the spacecraft projected to the equatorial plane
        at the times of the electric field measurements
    E : `xarray.DataArray`
        Electric field data
    L_sc : `xarray.DataArray`
        Geocentric distance to the spacecraft projected to the equatorial plane
        at times throughout the orbit. Provides context for when `E` is not sampled
        at all times
    title : str
        Plot title
    cblabel : str
        Label for the colorbar (electric field)

    Returns
    -------
    fig : `plt.figure`
        A figure object
    axes : list of `plt.subplots.axes`
        Subplots axes objects
    '''

    # Create the figure
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        fig = ax.get_figure()

    # Plot |E| on L vs. t plot
    cm = plt.cm.get_cmap('viridis')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    axlim = L.max()

    # Plot the orbit below the electric field
    if L_sc is not None:
        ax.plot(L_sc['time'], L_sc)
        axlim = max(axlim, L_sc.max())

    # Plot the electric field
    psc = ax.scatter(E['time'].data, L,
                     c=np.linalg.norm(E, ord=2, axis=1), cmap=cm)

    # Annotate plot
    ax.set_title(title)
    ax.set_ylabel('L')
    ax.set_ylim([1, 0.5 * np.ceil(axlim / 0.5)])
    format_axes(ax)

    # Create a colorbar
    cbar = add_colorbar(ax, psc)
    #    cbar = plt.colorbar(psc)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(cblabel, rotation=270)

    return fig, ax


def plot_orbit_coverage(R_sc, R, E, ax=None, title=''):
    '''
    Plot electric field data along the spacecraft orbit.

    Parameters
    ----------
    R_sc : `xarray.DataArray`
        Position of the spacecraft
    R : `xarray.DataArray`
        Position of the spacecraft at the times when there is electric field data
    E : `xarray.DataArray`
        Electric field data

    Returns
    -------
    fig : `plt.figure`
        A figure object
    axes : list of `plt.subplots.axes`
        Subplots axes objects
    '''
    axlim = np.linalg.norm(R_sc, ord=2, axis=1).max() / R_E
    axlim = 0.5 * np.ceil(axlim / 0.5)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        fig = ax.get_figure()

    # Plot the orbit
    ax.set_box_aspect(1)

    # Plot the orbit
    ax.plot(R_sc[:, 0] / R_E, R_sc[:, 1] / R_E)
    ax.set_title(title)
    ax.set_xlim([-axlim, axlim])
    ax.set_xlabel('X ($R_{E}$)')
    ax.set_ylim([-axlim, axlim])
    ax.set_ylabel('Y ($R_{E}$)')

    # Overlay where the data was taken
    if R_E is not None:
        ax.scatter(R[:, 0] / R_E, R[:, 1] / R_E, color='r')

    # Put an Earth at the center
    draw_earth_cart(ax)

    return fig, ax


def plot_orbit_quiver(R_sc, R, E, ax=None, title=''):
    '''
    Plot electric field data along the spacecraft orbit.

    Parameters
    ----------
    R_sc : `xarray.DataArray`
        Position of the spacecraft
    R : `xarray.DataArray`
        Position of the spacecraft at the times when there is electric field data
    E : `xarray.DataArray`
        Electric field data

    Returns
    -------
    fig : `plt.figure`
        A figure object
    axes : list of `plt.subplots.axes`
        Subplots axes objects
    '''
    axlim = np.linalg.norm(R_sc, ord=2, axis=1).max() / R_E
    axlim = 0.5 * np.ceil(axlim / 0.5)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        fig = ax.get_figure()

    # Quiver plot of E-field
    ax.set_box_aspect(1)

    # Plot the E-field
    ax.plot(R_sc[:, 0] / R_E, R_sc[:, 1] / R_E)
    ax.quiver(R[:, 0] / R_E, R[:, 1] / R_E, E[:, 0], E[:, 1], color='g')
    ax.set_title(title)
    ax.set_xlim([-axlim, axlim])
    ax.set_xlabel('X ($R_{E}$)')
    ax.set_ylim([-axlim, axlim])
    ax.set_ylabel('Y ($R_{E}$)')

    # Put an earth at the center
    draw_earth_cart(ax)

    return fig, ax


def plot_orbit(R_sc, R, E, ax=None):
    '''
    Plot electric field data along the spacecraft orbit.

    Parameters
    ----------
    R_sc : `xarray.DataArray`
        Position of the spacecraft
    R : `xarray.DataArray`
        Position of the spacecraft at the times when there is electric field data
    E : `xarray.DataArray`
        Electric field data

    Returns
    -------
    fig : `plt.figure`
        A figure object
    axes : list of `plt.subplots.axes`
        Subplots axes objects
    '''
    axlim = np.linalg.norm(R_sc, ord=2, axis=1).max() / R_E
    axlim = 0.5 * np.ceil(axlim / 0.5)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=False)
    else:
        fig = ax.get_figure()

    # Plot the orbit
    ax = axes[0, 0]
    ax.set_box_aspect(1)

    # Plot the orbit
    ax.plot(R_sc[:, 0] / R_E, R_sc[:, 1] / R_E)
    ax.set_title('Data along Orbit Track')
    ax.set_xlim([-axlim, axlim])
    ax.set_xlabel('X ($R_{E}$)')
    ax.set_ylim([-axlim, axlim])
    ax.set_ylabel('Y ($R_{E}$)')

    # Overlay where the data was taken
    if R_E is not None:
        ax.scatter(R[:, 0] / R_E, R[:, 1] / R_E, color='r')

    # Put an Earth at the center
    draw_earth_cart(ax)

    # Quiver plot of E-field
    ax = axes[0, 1]
    ax.set_box_aspect(1)

    # Plot the E-field
    ax.plot(R_sc[:, 0] / R_E, R_sc[:, 1] / R_E)
    ax.quiver(R[:, 0] / R_E, R[:, 1] / R_E, E[:, 0], E[:, 1], color='g')
    ax.set_title('$E$ along Orbit')
    ax.set_xlim([-axlim, axlim])
    ax.set_xlabel('X ($R_{E}$)')
    ax.set_ylim([-axlim, axlim])
    ax.set_ylabel('Y ($R_{E}$)')

    # Put an earth at the center
    draw_earth_cart(ax)

    plt.subplots_adjust(wspace=0.6)

    return fig, axes


def plot_vector_ts(axes, E, scatter=False, label='', title='', color=None):
    '''
    Plot a vector time series.

    Parameters
    ----------
    axes : list of axes objects
        Three axes objects in which to place the three components of the vector
    E : `xarray.DataArray`
        Vector time series
    scatter : bool
        Create a scatter plot if True, or a line plot if False
    label : str
        Legend label
    title : str
        Plot title
    color : str
        Color of points or lines
    '''
    ax = axes[0]
    if scatter:
        ax.scatter(E['time'].data, E.loc[:, 'x'], label=label, color=color)
    else:
        ax.plot(E['time'].data, E.loc[:, 'x'], label=label, color=color)
    ax.set_title(title)
    ax.set_ylabel('Ex\n(mV/m)')
    format_axes(ax, xaxis='off')

    ax = axes[1]
    if scatter:
        ax.scatter(E['time'].data, E.loc[:, 'y'], label=label, color=color)
    else:
        ax.plot(E['time'].data, E.loc[:, 'y'], label=label, color=color)
    ax.set_ylabel('Ey\n(mV/m)')
    format_axes(ax, xaxis='off')

    ax = axes[2]
    if scatter:
        ax.scatter(E['time'].data, E.loc[:, 'z'], label=label, color=color)
    else:
        ax.plot(E['time'].data, E.loc[:, 'z'], label=label, color=color)
    ax.set_ylabel('Ez\n(mV/m)')
    format_axes(ax)