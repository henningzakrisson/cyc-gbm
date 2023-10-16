# cyc-gbm
A package for the Cyclical Gradient Boosting Machine algorithm. For the (pre-print) paper describing the algorithm, see [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4352505).

## Installation
You can install the package using pip:
```bash
pip install cyc-gbm
```
Alternatively, you can install the package from source.
This will also include a pipeline for reproducing the results in the paper. Follow these steps:

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/henningzakrisson/cyc-gbm.git
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
   learning_rate=0.1,
   n_estimators=[26, 34],
   min_samples_split = 2,
   min_samples_leaf=20,
   max_depth=2,

)
model.fit(X_train, y_train)

# Evaluate
loss = model.dist.loss(y=y_test, z=model.predict(X_test)).sum()
print(f'negative log likelihood: {loss}')
```

## Contact
If you have any questions, feel free to contact me [here](mailto:henning.zakrisson@gmail.com).

