"""Cluster population-state vectors into discrete behavioral states."""

from sklearn.cluster import KMeans


def cluster_states(features, k=4):
    """K-Means clustering of population-state vectors.

    Why K-Means, and why k = 4?
    ---------------------------
    K-Means is the simplest unsupervised method that returns hard cluster
    labels - a good baseline before reaching for HDBSCAN, Gaussian mixtures,
    or HMMs. The intuition for using it here: if behavioral states form
    well-separated clouds in firing-rate space (the neural manifold
    hypothesis), then minimizing within-cluster variance should recover
    those states without any temporal modeling.

    k = 4 mirrors the number of simulated behavioral states (Rest, Explore,
    Active, Groom). Biologically, k is a hyperparameter that should be
    chosen by the experimenter to match the behavioral coarseness of
    interest, or by a model-selection criterion (silhouette, BIC, gap
    statistic) when ground truth is unavailable.

    Parameters
    ----------
    features : ndarray (n_timesteps, n_neurons)
        Smoothed firing-rate vectors. Note that we cluster in the
        *original* high-D space rather than in the 2-D embedding - UMAP
        warps distances nonlinearly, so clustering its output would
        confound geometric distortion with population state.
    k : int
        Number of clusters / latent behavioral states.

    Returns
    -------
    cluster_labels : ndarray (n_timesteps,), int
        Cluster index in [0, k) for each timestep.
    """
    # n_init=10 restarts to avoid a bad local optimum from a single random
    # initialization. random_state=42 makes the assignment reproducible
    # across runs (cluster *indices* are arbitrary, but the partition is
    # deterministic).
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    return model.fit_predict(features)
