# PSI-Net: Protocol-Segmented Inception Network for Leakage-Audited IIoT Intrusion Detection

Code for the manuscript *Protocol-Segmented Inception Network with Connection-Level Conformal Calibration for Leakage-Audited Intrusion Detection in Industrial IoT*.

All files sit in one folder. Running the experiments creates `out/` (prepared data, per-fold results) and `figs/`.

## What is here

| File | Purpose |
|---|---|
| `prep_data.py` | Identifier audit, protocol segments, connection/session group keys for Edge-IIoTset and WUSTL-IIoT-2021 |
| `psi_lib.py` | Loading, connection-disjoint folds, signed-log scaling, piecewise-linear codes, Fiedler seriation, metrics, label-conflict ceiling |
| `psi_models.py` | PSI-Net (field tokenizer, masked PS-Inception blocks, presence-weighted pooling) and all neural baselines in JAX/Flax, plus the trainer |
| `run_exp.py` | One experiment per (dataset, protocol, model, fold); writes metrics, test probabilities and embeddings |
| `conformal_eval.py`, `conf_groups.py` | Connection-level and record-level Mondrian conformal tests, threshold baselines |
| `aggregate.py`, `stats_tests.py` | Pooled out-of-fold metrics, connection bootstrap, Friedman/Wilcoxon/Holm |
| `make_all.py` | Collects every number used in the paper into `out/V.json` |
| `fig_arch.py`, `fig_results.py`, `fig_extra.py` | Grayscale figures |
| `timing.py` | CPU inference throughput and latency |
| `run_all.sh` | Full reproduction script |
| `summary_results.json` | The `V.json` produced for the paper |
| `results_per_fold.json` | Every per-fold result file of the paper runs |
| `timing.json` | Measured inference cost |
| `meta_*.json` | Feature lists, protocol segments and class counts of the three prepared views |

## Data

* Edge-IIoTset (ML release, `ML-EdgeIIoT-dataset.csv`): IEEE DataPort, http://ieee-dataport.org/8939. A mirror was used: `https://huggingface.co/datasets/Sunayanajagadesh/ML_EdgeIIoT_dataset`.
* WUSTL-IIoT-2021 (`wustl_iiot_2021.csv`): IEEE DataPort, doi:10.21227/yftq-n229. A Git LFS mirror was used: `https://github.com/sofia-on-air/wustl_iiot_2021`.

Place both CSV files in a folder (default `raw/`).

## Running

```bash
pip install -r requirements.txt
./run_all.sh raw        # about 10 hours on two CPU cores
```

Every neural model is trained for at most 12 epochs and 150 s per fold on CPU; the random seed equals the fold index. Model names: `psi+ple` (PSI-Net), `inc1d`, `inc2d`, `cnn1d`, `cnnbigru`, `mlp`, `ftt+ple`, `lgbm`, `xgb`, `rf`, `lr`; ablation suffixes `+nomask`, `+noseg`, `+nocross`, `+randin`; `tokinc+ple`; `inc1d+ser`, `inc1d+permN`.

## Licence

MIT, see `LICENSE`.
