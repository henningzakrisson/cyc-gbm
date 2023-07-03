# cyc-gbm
A package for the Cyclical Gradient Boosting Machine algorithm. For the (pre-print) paper describing the algorithm, see [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4352505).

## Installation
You can install the package using pip:
```bash
pip install cyc-gbm
```
Alternatively, you can install the package from source.
This will also include scripts for reproducing the results in the papar. Follow these steps:

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
   kappa=[26, 34],
   eps=0.1,
   max_depth=2,
   min_samples_leaf=20,
)
model.fit(X_train, y_train)

# Evaluate
loss = model.dist.loss(y=y_test, z=model.predict(X_test)).sum()
print(f'negative log likelihood: {loss}')
```

## Reproducing the numerical illustrations in the paper
The numerical illustrations in the paper can be reproduced by running the ````numerical_illustration```` function in the ````numerical_illustration/numerical_illustration.py```` module. 
The function takes the path to a configuration file as input. 
The configuration file is a yaml file that specifies the parameters of the numerical illustration.
An example configuration file can be found in ````numerical_illustration/config/simulation_config.yaml````.
For running several experiments in one run, I refer to the ````numerical_illustrations```` function in the same module. 
See the documentation for usage.
An example configuration file for running several experiments can be found in ````numerical_illustration/config/simulation_run/master_config.yaml````.

## Not yet implemented
- [ ] Add support for categorical features (currently the trees are based on ````sklearn.tree.DecisionTreeRegressor```` which does not support categorical features)
- [ ] Add other tuning methods (such as adaptive shrinkage)

## Contact
If you have any questions, feel free to contact me [here](mailto:henning.zakrisson@gmail.com).

