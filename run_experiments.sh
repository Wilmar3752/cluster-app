echo 'Setting all experiments'
dvc exp run -S 'train.imputation_method=median,mean' \ 
            -S 'train.scaler_method=standard,minmax' \
            -S 'train.n_clusters=3,4,5,6' \
            --queue
echo 'Running all queued experiments'
dvc exp run --run-all --jobs 4
echo 'Show all experiments by silhouette'
dvc exp show --sort-by Silhouette_score --sort-order desc