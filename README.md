# anes

To download all dependencies and compile the Rust program:

  `cargo b --release`

To run it and see all available options:

   `target/release/anes -h`
  
Options are the following

    -h,--help             Show this help message and exit
    -f,--file FILE        Name of the file (without extension) to load (Default: one_week)
    -c,--csv              Use csv format instead of parquet format (default: false)
    -n,--name NAME        Name of the variable to process (Default: pam)
    -p,--proba PROBA      Percentage of the dataset used for learning (Default: 0.8)
    -b,--before BEFORE    Number of values before the value to predict (Default: 5)
    -a,--after AFTER      Number of values after the value to predict (Default: 5)
    -f,--fsigma FSIGMA    Sigma multiplier for select (Default: 2.0)
    -l,--LocalSelect      Use local sigma for select (default: Global)
    -g,--GlobalNormalize  Use global normalization (default: Local)
    -t,--TaintedNormalize Use tainted normalization (default: Local)
    -s,--seed SEED        Seed of random number generator (Default: 0)

`target/release/anes -g -a 3 -b 3 -p 0.8 -f 3.0`  : source file is `one_week.parquet`, the variable observed is `pam`, normalization is done globally (mean and sigma are computed on the whole set and not only on each patient). We use 3 values before the value to predict and 3 values after the value to predict. The learning dataset is 80% of the whole dataset. Values above or below m+/-3sigma (here computed globally also) are considered as outliers and are supressed from the set. The program also computes using a least square minimization method the best coefficients x such that ||Ax-b|| is minimal, where A is the matrix in `source_learn` and b is the vector in `obj_learn`. It also computes the RMSE obtained when using x for predictions, both on the learning set and on the test set. The program produces four files named `source_learn.txt`, `obj_learn.txt`, `source_test.txt`, and `obj_test.txt`. These files can be directly used by the python program `anes.py` which will try to predict `obj` from `source` using simple neural networks.

You need Rust to be installed (see [Rust Install](https://www.rust-lang.org/tools/install)), and pytorch if you want to use `anes.py` (preferred method: [install miniconda](https://www.anaconda.com/download/success) and then run `conda install pytorch pytorch-cuda torchvision torchaudio -c pytorch -c nvidia`. If you don't have an nvidia card, remove pytorch-cuda from the previous list).
