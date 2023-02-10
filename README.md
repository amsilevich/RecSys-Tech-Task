## Recommendation System

`recsys` `ALS` `SVD` `ml`


### Description
This is library, which is a recommendation system that can recommend items to users based on their preferences as well as the preferences of other users.

### Algorithms
1. ALS (Alternating Least Square) algorithm - Popular collaborative filtering algorithm with the O(mr^2 + r^3) asymptotic.

### How to run
1. Install all packages from requirements.txt
```commandline
pip install -r requirements.txt
```
2. You need to have at least **Interactions** matrix, which describes User-Item interactions.
3. Write config for training pipeline or use default config (configs folder)
4. Run main script
```commandline
PYTHONPATH=. python3 src/run.py --data ./path/to/data --config  ./path/to/config
```
