## Deep Kernel Survival Analysis and Subject-Specific Survival Time Prediction Intervals

Author: George H. Chen (georgechen [at symbol] cmu.edu)

This code accompanies the paper:

> George H. Chen. "Deep Kernel Survival Analysis and Subject-Specific Survival Time Prediction Intervals". MLHC 2020.

Code requirements:

- Anaconda Python 3 (tested with Python version 3.6)
- Additional packages: joblib, lifelines, pyarrow, pytorch (tested with PyTorch version 1.5 with CUDA 10.2)
- cython compilation is required for the random survival forests implementation used:

```
python setup_random_survival_forest_cython.py build_ext --inplace
```

Note that this code comes with slightly modified versions of Haavard Kvamme's [pycox](https://github.com/havakv/pycox) and [torchtuples](https://github.com/havakv/torchtuples) packages (some bug fixes/print flushing).

I only tested the code in Ubuntu. The experiments in the paper were run on an Amazon p3.2xlarge instance using the Deep Learning (Ubuntu 18.04) Version 30.0 AMI.

The main code for neural kernel survival estimators from the paper is in `neural_kernel_survival.py`.

I've also included a few publicly available datasets, which can be loaded via `load_dataset` from `survival_datasets.py`, where to the extent possible I retained the original feature names, which could potentially be helpful for interpretation purposes:

- The SUPPORT dataset by Knaus et al 1995 (technically we use SUPPORT2) is taken from the [official page from Vanderbilt University](http://biostat.mc.vanderbilt.edu/wiki/Main/DataSets)
- The train/test split of the Rotterdam tumor bank (Foekens et al 2000) and German Breast Cancer Study Group (Schumacher et al 1994) datasets (train on Rotterdam and test on GBSG) is taken from the [DeepSurv (Katzman et al 2018) github repo](https://github.com/jaredleekatzman/DeepSurv)
- There are different versions of the METABRIC dataset (Curtis et al 2012) that are used; for simplicity we use the one that is built into the pycox package (this dataset is currently missing verified feature names)
- (Not part of my MLHC paper) There's also a recidivism dataset by Chung, Schmidt, and Witte (1991) that is included; some information is available [here](https://data.princeton.edu/pop509/recid1)

### Benchmarking methods based on concordance index

As an example of how to use the code, from within the base directory, you can train the NKS-Res-Diag model by running:

```
python benchmark/bench_nks_res_diag.py config.ini
```

The `config.ini` file contains hyperparameter search grids and other settings, including where output files should be saved (default: `./output/`).

After running the demo, in the output directory (default: `./output/`), you should find:

- `nks_res_diag_experiments1_cv5_test_metrics_bootstrap.csv` (contains final test set metrics, including time-dependent concordance index)
- `train/*_best_cv_hyperparams.pkl` (pickle files containing best hyperparameters found per dataset and also per experimental repeat if you are running with experimental repeats, which by default is turned off)
- `train/*_train_metrics.txt` (transcript that says what hyperparameter achieves what cross validation scores)
- `models/` (all trained models are saved here; re-running the demo will result in loading saved models rather than re-training them; to force re-training of a model, be sure to delete the corresponding saved models)
- `bootstrap/` (per model trained on the full training dataset after cross-validation, the test set bootstrap test metrics are stored in here)

Other methods (aside from NKS-Res-Diag) can be trained similarly (look in the `benchmark` folder to see all the other demo scripts). Note that random survival forests (`benchmark/bench_rsf.py`) must already be trained before training NKS-MLP with random survival forest initialization (`benchmark/bench_nks_mlp_init_rsf.py`). Similarly, DeepHit (`benchmark/bench_deephit.py`) must already be trained before training NKS-MLP with DeepHit initialization (`benchmark/bench_nks_mlp_init_deephit.py`).

After training all methods used in the paper, making the concordance index plot from the paper should be possible by running (note that the plot is saved by default to `./output/plots/`):

```
python visualization/plot_cindices.py config.ini survival_estimator_names.txt
```

Note that in `survival_estimator_names.txt`, you can comment out methods you want to exclude in the plot by putting `#` at the beginning of a method's line.

### Prediction intervals

After running the above demos, you can replicate the prediction interval experiments in the paper by running (using NKS-Res-Diag again as an example, on the metabric dataset for experiment 0--note that if you're not using experimental repeats the experiment number should always be 0):

```
python prediction_intervals/intervals_nks_res_diag.py config.ini 0 metabric
```

After running the demo, in the output directory (default: `./output/`), you should find:

- `split_conformal_prediction/*_qhats.txt` (marginal prediction interval radii -- multiply by 2 to get interval widths)
- `split_conformal_prediction/*_coverages.txt` (marginal prediction interval empirical coverage probabilities)
- `weighted_split_conformal_prediction/*_qhats.txt` (local prediction interval radii -- multiply by 2 to get interval widths)
- `weighted_split_conformal_prediction/*_coverages.txt` (local prediction interval empirical coverage probabilities)

After running the prediction interval demo scripts for all methods, making plots from the paper involving prediction intervals should be possible by running (plots are saved by default to `./output/plots/`):

```
python visualization/plot_interval_width_vs_coverage_marginal.py config.ini survival_estimator_names.txt
python visualization/plot_interval_width_vs_methods_marginal.py config.ini survival_estimator_names.txt
python visualization/plot_empirical_coverage_vs_calib_frac_marginal.py config.ini survival_estimator_names.txt
python visualization/plot_interval_width_vs_coverage_local.py config.ini survival_estimator_names.txt
```

### Running times

To obtain the cross-validation training time plots, after running the non-prediction-interval demos, run, for example:

```
python running_times/time_cv_nks_res_diag.py config.ini
```

This script aggregates timing data from cross-validation training and saves summary information to, by default, `./output/timing/*.pkl`.

Finally, after running the above for all methods, you shold be able to produce the timing plots from the paper via running:

```
python visualization/plot_cv_train_times_vs_methods.py config.ini survival_estimator_names.txt
python visualization/plot_n_durations_cv_train_times.py config.ini survival_estimator_names.txt
```
