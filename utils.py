import numpy as np
import matplotlib.pyplot as plt


def plot_learning(
    scores,
    filename,
    x=None,
    window=5,
    xlabel="Game",
    ylabel="Score",
    title=None,
    dpi=200,
    show=False,
):
    """
    Plot a running-average learning curve and save it to disk.

    Parameters
    ----------
    scores : array-like
        Sequence of scalar scores.
    filename : str
        Output path (e.g., "learning_curve.png").
    x : array-like or None
        X-axis values. If None, uses 0..N-1.
    window : int
        Running average window size (>=1).
    xlabel, ylabel : str
        Axis labels.
    title : str or None
        Optional plot title.
    dpi : int
        Saved figure DPI.
    show : bool
        If True, display the plot (plt.show()).

    Returns
    -------
    running_avg : np.ndarray
        The computed running average (length N).
    """
    scores = np.asarray(scores, dtype=float).ravel()
    n = scores.size
    if n == 0:
        raise ValueError("scores must be non-empty.")
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be an integer >= 1.")

    if x is None:
        x = np.arange(n)
    else:
        x = np.asarray(x)
        if x.shape[0] != n:
            raise ValueError("x must have the same length as scores.")

    # Efficient running average using cumulative sum:
    # avg[t] = mean(scores[max(0, t-window+1):t+1])
    csum = np.cumsum(np.insert(scores, 0, 0.0))
    running_avg = np.empty(n, dtype=float)
    for t in range(n):
        start = max(0, t - window + 1)
        running_avg[t] = (csum[t + 1] - csum[start]) / (t - start + 1)

    fig, ax = plt.subplots()
    ax.plot(x, running_avg)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)

    if show:
        plt.show()

    plt.close(fig)
    return running_avg
