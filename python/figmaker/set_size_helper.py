def set_size(width_pt=246.0, fraction=1.0, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width_pt : float
        Document width in points (e.g., \columnwidth=246.0 pt in APS REVTeX, \textwidth~510.0 pt).
    fraction : float
        Fraction of the width which you wish the figure to occupy.
    subplots : (rows, cols)
        The number of rows and columns of subplots.

    Returns
    -------
    fig_dim : tuple
        Dimensions of figure in inches.
    """
    # Width of figure
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch (TeX points)
    golden_ratio = (5**0.5 - 1) / 2  # aesthetic ratio
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


def set_size_ht(width_pt=246.0, fraction=1.0, subplots=(2, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width_pt : float
        Document width in points (e.g., \columnwidth=246.0 pt in APS REVTeX, \textwidth~510.0 pt).
    fraction : float
        Fraction of the height which you wish the figure to occupy.
    subplots : (rows, cols)
        The number of rows and columns of subplots.

    Returns
    -------
    fig_dim : tuple
        Dimensions of figure in inches.
    """
    # Width of figure
    fig_width_pt = width_pt 
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch (TeX points)
    golden_ratio = (5**0.5 - 1) / 2  # aesthetic ratio
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * fraction * subplots[0]/subplots[1] 
    return (fig_width_in, fig_height_in)

def align_x_axis(ax_ref, ax_target):
    """
    Align the x-axis of ax_target to ax_ref:
    - scale (linear/log/etc.)
    - limits
    - tick locator (major + minor)
    - tick formatter (major + minor)

    Assumes x-quantities differ but visual alignment is desired.
    """

    # --- scale & limits ---
    ax_target.set_xscale(ax_ref.get_xscale())
    ax_target.set_xlim(ax_ref.get_xlim())

    # --- major ticks ---
    ax_target.xaxis.set_major_locator(ax_ref.xaxis.get_major_locator())
    ax_target.xaxis.set_major_formatter(ax_ref.xaxis.get_major_formatter())

    # --- minor ticks ---
    ax_target.xaxis.set_minor_locator(ax_ref.xaxis.get_minor_locator())
    ax_target.xaxis.set_minor_formatter(ax_ref.xaxis.get_minor_formatter())

    # --- log-specific housekeeping ---
    if ax_ref.get_xscale() == "log":
        ax_target.minorticks_on()

