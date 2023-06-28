# cyc-gbm
A package for the Cyclical Gradient Boosting Machine algorithm. For the (pre-print) paper describing the algorithm, see [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4352505).

## Installation
1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/henningzakrisson/c-gbm.git
    ```
2. Create a virtual environment in the root directory of the repository:
    ```bash
    python3 -m venv venv
    ```
3. Activate the virtual environment:
    ```bash
    source venv/bin/activate
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
## Usage example
Fitting the mean and (log) sigma parameters of a normal distribution to a simulated dataset:

```python
import numpy as np
from cyc_gbm import CycGBM
from cyc_gbm import initiate_distribution

# Simulate data
X = np.random.normal(size=(10000, 3))
mu = X[:, 0] + 10 * (X[:, 1] > 0)
log_sigma = 3 - 2 * (X[:, 1] > 0)
z = np.stack([mu, log_sigma], axis=0)
dist = initiate_distribution(distribution='normal')
y = dist.simulate(z=z)

# Split data
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
idx_train, idx_test = idx[:500], idx[500:]
X_train, y_train, z_train = X[idx_train], y[idx_train], z[:, idx_train]
X_test, y_test, z_test = X[idx_test], y[idx_test], z[:, idx_test]

# Fit model
model = CycGBM(
   distribution='normal',
   kappa=[40, 20],
   eps=0.1,
   max_depth=2,
   min_samples_leaf=10,
)
model.fit(X_train, y_train)

# Evaluate
z_hat = model.predict(X_test)
loss = model.dist.loss(y=y_test, z=z_hat).sum()
print(f'negative log likelihood: {loss}')
```

## Reproducing the numerical illustrations in the paper
The numerical illustrations in the paper can be reproduced by running the ````numerical_illustration```` function in the ````src/numerical_illustration.py```` module. 
The function takes the path to a configuration file as input. 
The configuration file is a yaml file that specifies the parameters of the numerical illustration.
An example configuration file can be found in ````config/simulation_config.yaml````.
For running several experiments in one run, I refer to the ````numerical_illustrations```` function in the same module. 
See the documentation for usage.
An example configuration file for running several experiments can be found in ````config/simulation_run/master_config.yaml````.

## Contact
If you have any questions, feel free to contact me [here](mailto:henning.zakrisson@gmail.com).

