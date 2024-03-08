import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from cubic_regular.cubicmin import cubic_regularization, cubic_minimization
import torch

class CubicFit(BaseEstimator, ClassifierMixin):
    def __init__(self, L, kappa, lr, momentum, dampening):
        self.L = L
        self.kappa = kappa
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
    def fit(self, params, target):
        self.X_ = params
        self.y_ = target
        self.classes_ = [1]
        # Return the classifier
        return self
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        grad_vec, gnorm = X[0][:3], np.linalg.norm(X[:3])
        hessian, hnorm = X[0][3:].reshape(3,3), np.linalg.norm(X[3:])
        res= None

        initial_vector = torch.zeros(hessian.shape[0])
        optimizer = torch.optim.SGD([initial_vector], lr=self.lr)
        optargs = {'params': [initial_vector], 
					'lr': self.lr, 
					'weight_decay': 0,
					'momentum': self.momentum,
					'nesterov': True, 
					'maximize': False,
					'foreach': None,
					'dampening': self.dampening,
					'differentiable': False}

        res, _ = cubic_minimization(grad=grad_vec, gnorm=gnorm, 
			hessian=hessian, hnorm=hnorm, L=self.L, kappa=self.kappa, 
			optimizer=optimizer, tol=0.001, max_iterno=100,
			check=True, **optargs)

        # Calculate cubic regularization function for returned vectors
        reg_value = cubic_regularization(grad_vec, hessian, res[1], self.L)	
        # Check if there is a lowest eigenvector approximation
        if res[2] is not None:
            reg_value_min = res[0]/(2*self.L)*cubic_regularization(
                grad_vec, hessian, res[2], self.L)
            # Keep the vector that gives smaller regular/tion value
            if reg_value_min<reg_value:
                reg_value = reg_value_min
        return [reg_value]

X = [np.random.randn(12,) for _ in range(5)]
y = np.array([0 for _ in range(5)])

cb = CubicFit(1, 1, 0.1, 0.1, 0.1)
# print(cb)
# cb.fit(X, y)
# print(cb.predict(X))

# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingRandomSearchCV
rng = np.random.RandomState(0)
param_dist = {
    "L": [1, 10], 
    "kappa": [x for x in range(30, 35)], 
    "lr": [0.1, 1], #[1e-5, 1e-3, 1e-2, 1e-1, 1], 
    "momentum": [x/10 for x in range(1,2)], 
    "dampening": [x/10 for x in range(1,2)],
}
rsh = HalvingRandomSearchCV(
    estimator=cb, param_distributions=param_dist, 
    factor=2, random_state=rng, max_resources=6,
    cv=2, scoring='accuracy'
)
rsh.fit(X, y)
