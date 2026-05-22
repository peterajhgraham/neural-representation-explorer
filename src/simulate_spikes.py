"""Spike-train simulator for a population of state-tuned neurons.

The generative model is a deliberately minimal caricature of cortical
population activity: a Markov chain over discrete behavioral states drives
state-tuned Poisson firing in overlapping ensembles. The point is to create
data with *genuine* low-dimensional manifold structure — distinct states
should occupy distinct regions of neural-activity space — so that downstream
dimensionality-reduction tools have something real to recover.
"""

import numpy as np


def simulate_spikes(n_neurons=60, n_timesteps=2000, n_states=4, seed=42):
    """Simulate structured neural population activity with behavioral states.

    Each behavioral state strongly activates a dedicated *ensemble* of
    neurons — overlapping subpopulations whose joint firing pattern is the
    neural signature of that state. This mirrors how real cortical neurons
    are organized into task- or stimulus-tuned assemblies (Hebbian ensembles
    in the classical literature; "cell assemblies" in Buzsáki's framing).
    State transitions follow a Markov chain with exponentially-distributed
    dwell times, a standard model for spontaneous brain-state dynamics.

    Parameters
    ----------
    n_neurons : int
        Number of simulated units. 60 is small enough to plot easily and
        large enough that the manifold structure is non-trivial.
    n_timesteps : int
        Length of the recording in bins (treat each bin as ~10 ms, so 2000
        bins ≈ 20 s of recording — comparable to a short behavioral epoch).
    n_states : int
        Number of distinct behavioral states. With k=4 we get Rest /
        Explore / Active / Groom, mirroring coarse behavioral-state
        ethograms used in head-fixed mouse experiments.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    spikes : ndarray (n_neurons, n_timesteps), float32
        Poisson spike counts per bin.
    state_labels : ndarray (n_timesteps,), int
        Ground-truth behavioral-state index at each timestep.
    """
    rng = np.random.default_rng(seed)

    # Heterogeneous baseline rates across the population.
    # Real cortical units span ~3 orders of magnitude in firing rate; we use
    # a narrow uniform band here to keep the toy simulation tractable.
    baseline_firing_rates = rng.uniform(0.02, 0.06, size=n_neurons)

    # Each state strongly activates a contiguous ensemble. Contiguity has no
    # biological meaning — it just makes the firing-rate heatmap legible.
    ensemble_size = n_neurons // n_states
    state_to_neuron_rates = np.tile(baseline_firing_rates, (n_states, 1))
    for state_idx in range(n_states):
        start = state_idx * ensemble_size
        end = start + ensemble_size
        # 5–9× rate gain inside the active ensemble. This factor controls
        # how separable the states will be downstream; lower values would
        # produce more overlapping manifolds (and a more realistic SNR).
        state_to_neuron_rates[state_idx, start:end] *= rng.uniform(5.0, 9.0, size=ensemble_size)

    # Markov state sequence — uniform off-diagonal transitions so any state
    # can follow any other. Dwell times are sampled directly (semi-Markov),
    # giving longer, more realistic state durations than a vanilla
    # one-step-Markov chain would.
    transition_matrix = np.full((n_states, n_states), 1.0 / (n_states - 1))
    np.fill_diagonal(transition_matrix, 0.0)

    state_labels = np.zeros(n_timesteps, dtype=int)
    t, current_state = 0, 0
    while t < n_timesteps:
        # Exponential dwell-time draw clipped to [50, 400] bins. The mean
        # (~150 bins ≈ 1.5 s) sits in the range observed for behavioral
        # macro-states in rodents — long enough to be obvious in a raster.
        dwell = int(np.clip(rng.exponential(150), 50, 400))
        end = min(t + dwell, n_timesteps)
        state_labels[t:end] = current_state
        current_state = rng.choice(n_states, p=transition_matrix[current_state])
        t = end

    # Inhomogeneous Poisson process: rate switches with state, spikes are
    # otherwise independent in time. This matches the textbook "rate-coding"
    # null model and is what PCA/UMAP downstream are implicitly built for.
    per_timestep_rates = state_to_neuron_rates[state_labels].T  # (n_neurons, T)
    spikes = rng.poisson(per_timestep_rates).astype(np.float32)

    return spikes, state_labels
