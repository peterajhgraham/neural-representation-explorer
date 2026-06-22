# Neural Representation Explorer

**60 simulated neurons, 4 behavioral states. PCA and UMAP recover the full state structure in 2-D, with no labels.**

The population activity of 60 neurons lives in a 60-dimensional space, but the *behaviorally relevant* signal collapses onto a 2-D manifold cleanly enough that unsupervised K-Means (k=4) finds every state. Silhouette ≈ 0.53; 13 PCs capture 90% of the variance. The neural manifold hypothesis, on a toy.

![Manifolds](results/manifolds.png)

> [!NOTE]
> The pipeline auto-runs on every push via GitHub Actions and commits the refreshed figures back to the repo, so the images here always reflect the latest code.

## How it works

The pipeline simulates 60 neurons firing over 2000 timesteps. Each neuron's rate is set by whichever of four behavioral states (Rest, Explore, Active, Groom) is currently active: state-tuned ensembles fire as Poisson processes, and the state sequence is a Markov chain with exponential dwell times. A Gaussian kernel (σ = 10 bins) smooths the raw spikes into firing-rate vectors. PCA (linear, global) and UMAP (nonlinear, local) compress the 60-D activity to 2-D, and K-Means labels each timestep with its inferred state.

The result worth noting is *not* that K-Means works, but that the 2-D embedding has enough structure for it to work at all. Across motor cortex, prefrontal cortex, and hippocampus, real population activity is confined to a manifold whose dimensionality tracks the number of behaviors, not the number of neurons. This is the minimal version of that picture: distinct states sit in distinct regions of a low-D embedding, and the trajectory through state space is recoverable without supervision.

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py
# → results/ contains all figures and summary.json
```

## Figures

| Figure | What it shows |
|--------|---------------|
| `manifolds.png` | PCA & UMAP scatter, colored by K-Means cluster |
| `trajectory.png` | Population path through state space; colored by time (left) and ground-truth state (right) |
| `spike_raster.png` | Raw spikes for 40 neurons, annotated with behavioral state |
| `firing_rates.png` | Gaussian-smoothed activity heatmap; bright bands mark ensemble activation |
| `transitions.png` | Empirical Markov transition matrix |
| `pca_variance.png` | Per-component and cumulative explained variance |

Full metrics: [`results/RESULTS.md`](results/RESULTS.md).

![Trajectory](results/trajectory.png)
![PCA Variance](results/pca_variance.png)

## Extending to real data

The pipeline accepts real extracellular recordings via `--mode real`, which streams a public NWB file from the [DANDI Archive](https://dandiarchive.org/) and bins spike times into the same `(n_neurons, n_timesteps)` array the simulator produces. Nothing downstream changes.

```bash
pip install dandi remfile h5py pynwb
python run_pipeline.py --mode real
```

The loader caps the streamed window at 20 s (≈ 2000 bins at 10 ms) so you can iterate without pulling gigabytes. To point it elsewhere, edit `DANDISET_ID` in `loaders/nwb_loader.py` or call `load_nwb_spikes("000409")` directly. Good starting dandisets: [`000003`](https://dandiarchive.org/dandiset/000003) (default), [`000409`](https://dandiarchive.org/dandiset/000409) (IBL Brain-Wide Map), [`000053`](https://dandiarchive.org/dandiset/000053) (Steinmetz et al. visual cortex). Recordings in other formats (Phy/Kilosort, Plexon, Blackrock) convert to NWB cleanly with [NeuroConv](https://neuroconv.readthedocs.io/).

Once `state_labels` is uninformative, a few outputs change meaning: clusters reflect *discovered* states rather than ground truth, the time-colored trajectory panel becomes the more useful one, and the silhouette score drops (often to 0.1–0.3) because real behavioral states overlap. The PCs-to-90% number is the headline: real cortex typically sits in the 10–30 range.

## Layout

```
run_pipeline.py        # end-to-end pipeline
src/                   # simulate · smooth · reduce · cluster
loaders/nwb_loader.py  # DANDI/NWB streaming for --mode real
notebooks/             # interactive exploration
results/               # auto-generated figures + summary.json
```

## References

The methods here lean on:

- Cunningham & Yu (2014), *Dimensionality reduction for large-scale neural recordings*, Nat Neurosci — the canonical review; the population is the computational unit, not the cell.
- Churchland et al. (2012), *Neural population dynamics during reaching*, Nature — motor cortex lives on a ~10-D manifold whose rotational dynamics predict muscle output.
- Gallego, Perich, Miller & Solla (2017), *Neural manifolds for the control of movement*, Neuron — manifolds are stable across days and partially conserved across animals.
- McInnes, Healy & Melville (2018), *UMAP*, arXiv:1802.03426.
