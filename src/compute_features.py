"""Convert binary/integer spike counts into continuous firing-rate features.

Spike trains are point processes - sparse, noisy, and discontinuous - which
makes them a poor input to PCA, UMAP, or any other Euclidean-geometry tool.
The standard fix is to convolve each neuron's spike train with a smoothing
kernel, producing a continuous firing-rate estimate at every timestep. This
is the "instantaneous firing rate" of the systems-neuroscience literature.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def compute_firing_rates(spikes, sigma=10):
    """Gaussian-kernel smoothed firing rates, vectorized.

    Why a Gaussian, and why σ = 10?
    -------------------------------
    A Gaussian kernel is the standard choice in neuroscience for converting
    spike counts into a smooth rate signal: it's symmetric (no temporal bias)
    and has compact effective support. The kernel width σ trades temporal
    resolution against noise:

    - σ too small → the rate trace stays spiky, PCA/UMAP latch onto Poisson
      noise instead of the behavioral signal.
    - σ too large → fast state transitions get smeared across boundaries
      and the manifold collapses.

    σ = 10 bins corresponds to ~100 ms at our default 10-ms bin size,
    which matches the timescale of cortical population dynamics during
    spontaneous behavior (a few hundred ms per behavioral state). This is
    the same order of magnitude used in Churchland et al. (2012) and most
    Neuropixels analyses of behavioral-state structure.

    Parameters
    ----------
    spikes : ndarray (n_neurons, n_timesteps)
        Binned spike counts.
    sigma : float
        Gaussian kernel standard deviation, in timesteps.

    Returns
    -------
    firing_rates : ndarray (n_timesteps, n_neurons)
        Smoothed firing rates, transposed so each row is a "population
        snapshot" - the natural input format for sklearn / UMAP.
    """
    # ±3σ captures >99.7% of the Gaussian's mass; truncating beyond that
    # is standard practice and keeps the kernel a manageable length.
    half_window = int(3 * sigma)
    x = np.arange(-half_window, half_window + 1, dtype=float)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()  # normalize so smoothing preserves the mean rate

    # Reflect-pad at the edges so the kernel doesn't bleed in zeros, which
    # would artificially depress the rate at the start and end of the trace.
    padded = np.pad(spikes, ((0, 0), (half_window, half_window)), mode="reflect")

    # sliding_window_view + elementwise multiply + sum = vectorized 1-D
    # convolution along the time axis. Faster and clearer than scipy here.
    windows = sliding_window_view(padded, len(kernel), axis=1)  # (n_neurons, T, K)
    smoothed = (windows * kernel).sum(axis=2)                   # (n_neurons, T)

    # Transpose so each row is one population state vector at one timestep.
    return smoothed.T
