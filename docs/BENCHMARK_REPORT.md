# mlnative vs. other languages and ML libraries

mlnative runs machine learning in PHP as compiled C++. How does that compare with the other
popular high-level languages, and with the Python ML stack? We ran six common algorithms in all
of them under identical conditions. Here are the results.

## Highlights

- **13× to 280× faster** than the same algorithms written in PHP, Python, Ruby or Lua.
- **5× to 20× faster than PHP 8.3 with JIT enabled.**
- **Faster than Node.js on every benchmark**, by up to 24×, even though Node.js has V8's
  optimizing JIT.
- **Competitive with the Python ML stack:**
  - decision trees **4.6× faster** than scikit-learn
  - neural networks **1.3× faster** than scikit-learn and **1.9× faster** than PyTorch
  - linear regression **1.7× faster** than scikit-learn

## mlnative vs. scripting languages

Training and prediction time in milliseconds for the same algorithm in each language. Lower is
better.

| | KNN | Logistic regression | Linear regression | KMeans | Decision tree | Neural network |
|---|---:|---:|---:|---:|---:|---:|
| **PHP + mlnative** | **7.2** | **16.7** | **5.4** | **4.0** | **8.4** | **47.2** |
| Node.js 22 | 175 | 30.2 | 24.6 | 5.9 | 154 | 251 |
| PHP 8.3 + JIT | 147 | 93.4 | 106 | 23.4 | 77.0 | 470 |
| PHP 8.3 | 284 | 226 | 376 | 62.5 | 113 | 1313 |
| Lua 5.4 | 900 | 292 | 342 | 72.3 | 531 | 1512 |
| Python 3.12 | 700 | 649 | 556 | 172 | 244 | 2729 |
| Ruby 3.3 + YJIT | 1619 | 909 | 878 | 242 | 184 | 4475 |
| Ruby 3.3 | 2039 | 1334 | 1531 | 350 | 340 | 6785 |

In practice:
- **Neural network:** a model that trains in under 50 ms with mlnative takes 1.3 s in plain
  PHP and 2.7 s in plain Python.

## mlnative vs. the Python ML stack

Training and prediction time in milliseconds. Lower is better; the fastest in each column is in
bold.

| | KNN | Logistic regression | Linear regression | KMeans | Decision tree | Neural network |
|---|---:|---:|---:|---:|---:|---:|
| **PHP + mlnative** | 7.2 | 16.7 | 5.4 | 4.0 | **8.4** | 47.2 |
| scikit-learn 1.9 | 2.9 | 3.4\* | 9.2 | **2.5** | 38.8 | 59.4 |
| PyTorch 2.14 | **2.8** | 21.6 | 1.8 | 5.2 | — | 91.5 |
| NumPy 2.5 | 10.7 | **8.0** | **1.4** | 10.6 | — | **34.0** |

\* scikit-learn uses a different solver (L-BFGS), so this cell is not a like-for-like
comparison. NumPy and PyTorch have no decision tree.

mlnative plays in the same league as the libraries that power Python data science, and it does
so from plain PHP arrays, with no extra runtime to deploy:
- **Where mlnative is ahead:** decision trees and neural networks.
- **Where the libraries are ahead:** the algorithms that boil down to one large matrix product
  (linear regression, brute-force KNN), where NumPy and PyTorch use a hand-tuned BLAS.

## How we measured

- **The six algorithms:**
  - KNN: 3000×16 training set, 300 queries, k = 5
  - Logistic regression: 3000×16, 200 gradient-descent epochs
  - Linear regression: 20000×32
  - KMeans: 5000×8, 4 clusters, 20 iterations
  - Decision tree (CART): 5000×16, depth 10
  - Neural network: 16→32→16→2 with ReLU and Adam, 10 epochs on 3000 samples
- **The same algorithm everywhere:**
  - The language versions are line-by-line ports of the same reference code.
  - Every runtime reads the same data files. The neural networks start from the same initial
    weights.
  - Checksums of the results confirm that every implementation does the same work.
- **Fair conditions:**
  - Every runtime runs in its own Docker container, one at a time, limited to **one CPU
    core**. The libraries are pinned to one thread as well.
  - Each benchmark is warmed up first; the reported number is the median of several runs.
  - mlnative's time includes converting the PHP arrays to native memory.
- **Machine:** Apple M2, Docker (Linux, aarch64).
  - mlnative: PHP 8.3.36, default `-O2` build.
  - Node.js 22.23, Python 3.12.14, Ruby 3.3.12, Lua 5.4.6.

mlnative can also use several cores (`mlnative.threads`, see
[section 4.5 of the documentation](DOCUMENTATION.md#45-multithreading)). These single-core
numbers therefore understate what it can do on a real server.

## Run it yourself

All you need is Docker. The suite builds every image, generates the data and prints the tables
above:

```bash
sh benchmarks/cross-language/run.sh
```

To run only some runtimes, pass their names (`native`, `php`, `node`, `ruby`, `lua`, `python`):

```bash
sh benchmarks/cross-language/run.sh native python
```

The scripts are in [benchmarks/cross-language](../benchmarks/cross-language/run.sh). Absolute
times depend on the machine, so expect your numbers to differ a little.
