import itertools

import subprocess

imputation_method_values = ['mean', 'median']
capping_method_values = ['gaussian', 'iqr', 'mad', 'quantiles']
tail_values = ['left', 'rigth', 'both']
scaler_method_values = ['standard', 'mimnax', 'none']
n_components_values = [2,3,4,5]
n_clusters_values = [1,2,3,4,5,6]

for im, cm, tail, sm, nco, ncl in itertools.product(imputation_method_values, 
                                                    capping_method_values,
                                                    tail_values,
                                                    scaler_method_values,
                                                    n_components_values,
                                                    n_clusters_values):
    subprocess.run(["dvc", "exp", "run", "--queue",
                    "--set-param", f"train.imputation_method={im}",
                    "--set-param", f"train.scaler_method={sm}",
                    "--set-param", f"train.n_components={nco}",
                    "--set-param", f"train.n_clusters={ncl}",
                    "--set-param", f"train.capping_method={cm}",
                    "--set-param", f"train.tail={tail}"])

