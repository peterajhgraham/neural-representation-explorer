"""Dimensionality-reduction methods for neural population state vectors.

PCA gives the linear, global view: the axes of greatest variance through
the population activity, which in practice line up with the
behaviorally-relevant manifold whenever state-tuned ensembles dominate the
covariance structure. UMAP gives the nonlinear, local view: it preserves
neighborhood relationships in the high-D space, which is often what you
want when the manifold is curved or when states form genuinely separated
clusters (the typical case for behavioral-state data).

Using both is standard practice — if PCA and UMAP agree on the cluster
structure, that's strong evidence the structure is in the data and not an
artifact of either method.
"""

from sklearn.decomposition import PCA
import umap


def compute_pca(features, n_components=2):
    """Project population state vectors onto their top principal components.

    PCA finds orthogonal directions of maximum variance. For neural
    population data with state-tuned ensembles, these directions tend to
    align with the *behaviorally meaningful* axes simply because state
    switches are the dominant source of population-wide covariance.

    Parameters
    ----------
    features : ndarray (n_timesteps, n_neurons)
        Smoothed firing-rate vectors, one per timestep.
    n_components : int
        Number of principal components to keep. 2 for visualization;
        increase to estimate the manifold's effective dimensionality.

    Returns
    -------
    ndarray (n_timesteps, n_components)
        Population activity in PC space.
    """
    return PCA(n_components=n_components, random_state=42).fit_transform(features)


def compute_umap(features, n_neighbors=15, min_dist=0.1):
    """Nonlinear embedding via UMAP (McInnes et al., 2018).

    Where PCA captures global linear structure, UMAP captures local
    topology — it learns a low-D embedding that preserves which
    population-state vectors are *near* each other in high-D. For neural
    data this tends to make discrete behavioral states pop out as
    well-separated clusters even when they're not linearly separable.

    Parameters
    ----------
    features : ndarray (n_timesteps, n_neurons)
        Smoothed firing-rate vectors, one per timestep.
    n_neighbors : int
        Local-neighborhood size. Smaller → more local structure preserved
        (good for fine-grained substates); larger → more global structure.
        15 is the UMAP default and works well at this dataset size.
    min_dist : float
        Minimum spacing between embedded points. Smaller values make
        clusters tighter; 0.1 is a good default for visualization.

    Returns
    -------
    ndarray (n_timesteps, 2)
        2-D UMAP embedding of the population activity.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    return reducer.fit_transform(features)
