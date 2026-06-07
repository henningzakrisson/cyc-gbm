# cyc-gbm
A package for the Cyclic Gradient Boosting Machine algorithm. For the (pre-print) paper describing the algorithm, see [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4352505).

## Installation
You can install the package using pip:
```bash
pip install cyc-gbm
```
Alternatively, you can install the package from source.
This will also include a pipeline for reproducing the results in the paper:

```bash
git clone https://github.com/henningzakrisson/cyc-gbm.git
cd cyc-gbm
uv sync
```

To also install the numerical illustration dependencies:
```bash
uv sync --extra numerical-illustration
```

## Usage example
Fitting the mean and (log) sigma parameters of a normal distribution to a simulated dataset:

```python
import numpy as np
from cyc_gbm import CyclicalGradientBooster
from sklearn.model_selection import train_test_split

# Simulate data
X = np.random.normal(size=(1000, 2))
mu = X[:, 0] + 10 * (X[:, 1] > 0)
sigma = np.exp(3 - 2 * (X[:, 0] > 0))
y = np.random.normal(mu, sigma)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit model
model = CyclicalGradientBooster(
   distribution='normal',
   learning_rate=0.1,
   n_estimators=[26, 34],
   min_samples_split = 2,
   min_samples_leaf=20,
   max_depth=2,

)
model.fit(X_train, y_train)

# Evaluate
loss = model.distribution.loss(y=y_test, z=model.predict(X_test)).sum()
print(f'negative log likelihood: {loss}')
```

## Experimental: Categorical feature support

The base learner in `cyc-gbm` is scikit-learn's `DecisionTreeRegressor`, which
currently does not support native categorical splits. There is an
[open PR (scikit-learn #33354)](https://github.com/scikit-learn/scikit-learn/pull/33354)
that adds this functionality. **Once that PR is merged and released, this
workaround will no longer be required.**

To experiment with categorical features, install scikit-learn from the PR branch:

```bash
uv pip install "scikit-learn @ git+https://github.com/adam2392/scikit-learn.git@nocats-v2" --force-reinstall
```

> **Note:** This builds scikit-learn from source (C/Cython compilation) and
> requires a C compiler. On macOS this is included in the Xcode Command Line
> Tools (`xcode-select --install`); on Linux install `gcc`/`g++` from your
> package manager. The build takes roughly 5–15 minutes. To revert to the stable
> release, run:
> ```bash
> uv pip install scikit-learn==1.8.0 --force-reinstall
> ```

After installing the PR branch, use `.venv/bin/python` to run scripts — **not**
`uv run`. `uv run` re-syncs dependencies from `pyproject.toml` and will
downgrade scikit-learn back to `1.8.0`, undoing the workaround.

This is experimental and depends on an unmerged PR — the API may change.

## Contact
If you have any questions, feel free to contact me [here](mailto:henning.zakrisson@gmail.com).
