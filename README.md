## How to run the code

Everything needed to run the experiments shown in the paper is contained in the script `main.py`. 

A Conda `environment.yml` file is provided with all the dependencies needed to run the code. In order to install 
all the dependencies (assuming that Conda is already installed), just run `conda env create -f environment.yml`.
A new environment called `lip-std` will appeared, to activate it execute `conda activate lip-std`.

Running the algorithm is quite straightforward, a command line interface is provided and help is provided through
```bash
python main.py --help
```

As an example, we can run on the Breast dataset using the second mask that has 10% of missing data 
(located in dataset/Breast) by calling

```bash
python main.py -seed=7 -model=vae -dataset=datasets/Breast/ -miss-perc=10 -miss-suffix=2 -trick=gamma 
```

The previous command uses the Gamma trick (ours-gamma). Other options are `-trick=bern` for the Bernoulli trick
(ours-bern) and `-trick=none` to only treat continuous variables (ours-none).
Besides, standardization and normalization are supported via the `-std-none` and `-max` options. For example:

```bash
python main.py -seed=7 -model=vae -dataset=datasets/Breast/ -miss-perc=10 -miss-suffix=2 -std-none
```

For the sake of completeness, here is the output of the argument `--help`:
```buildoutcfg
usage:  [-h] [-seed SEED] [-root ROOT] [-to-file] [-batch-size BATCH_SIZE]
        [-learning-rate LEARNING_RATE] [-max-epochs MAX_EPOCHS]
        [-print-every PRINT_EVERY] -model {mm,vae,mf}
        [-latent-size LATENT_SIZE] [-num-clusters NUM_CLUSTERS]
        [-hidden-size HIDDEN_SIZE] -dataset DATASET -miss-perc MISS_PERC
        -miss-suffix MISS_SUFFIX [-trick {gamma,bern,none}] [-max] [-std-none]

optional arguments:
  -h, --help            show this help message and exit
  -seed SEED
  -root ROOT            Output folder (default: results)
  -to-file              Redirect output to 'stdout.txt'
  -batch-size BATCH_SIZE
                        Batch size (default: 1024)
  -learning-rate LEARNING_RATE
                        Learning rate (default: 1e-2 if MF, 1e-3 otherwise)
  -max-epochs MAX_EPOCHS
                        Max epochs (default: as described in the appendix)
  -print-every PRINT_EVERY
                        Interval to print (default: 25)
  -model {mm,vae,mf}    Model to use: Mixture Model (mm), Matrix Factorization
                        (mf), or VAE (vae)
  -latent-size LATENT_SIZE
  -num-clusters NUM_CLUSTERS
  -hidden-size HIDDEN_SIZE
                        Size of the hidden layers (VAE)
  -dataset DATASET      Dataset to use (path to folder)
  -miss-perc MISS_PERC  Missing percentage
  -miss-suffix MISS_SUFFIX
                        Suffix of the missing percentage file
  -trick {gamma,bern,none}
                        Trick to use (if any)
  -max                  Normalize data
  -std-none             Standardize data

```
