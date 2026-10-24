# ReFlow: Rethinking the Source and the Objective of Flow Matching for Collaborative Filtering

This repository contains the implementation for "ReFlow: Rethinking the Source and the Objective of Flow Matching for Collaborative Filtering" paper.

To reproduce the experiments, please follow the steps below:

1-Install the necessary packages using the following command:
```
conda env create --file requirements.yaml
```
<br><br><br>
2-For convenience and better reproducibility, all datasets are included in the data.zip file. Unzip them in the same directory as reflow.py script.
<br><br><br><br>
3-For each dataset, use the following commands to run the code:

Sports:
```
python reflow.py --train_path sportstrain.inter --valid_path sportsvalid.inter --test_path sportstest.inter --rank_lambda 0.1 --rank_num_neg 50 --rank_tau 0.5 --alpha 0.5 --kappa 10
```
Gowalla:
```
python reflow.py --train_path gowallatrain.inter --valid_path gowallavalid.inter --test_path gowallatest.inter --rank_lambda 0.01 --rank_num_neg 100 --rank_tau 0.2 --alpha 0.3 --kappa 2
```
Douban-Book:
```
python reflow.py --train_path doubantrain.inter --valid_path doubanvalid.inter --test_path doubantest.inter --rank_lambda 0.01 --rank_num_neg 500 --rank_tau 0.2 --alpha 0.5 --kappa 2
```
Yelp:
```
python reflow.py --train_path yelptrain.inter --valid_path yelpvalid.inter --test_path yelptest.inter --rank_lambda 0.01 --rank_num_neg 50 --rank_tau 0.2 --alpha 0.3 --kappa 5
```
