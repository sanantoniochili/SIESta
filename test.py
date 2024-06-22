import numpy as np
from relax.optim.cubic_minimization.main import CubicMin
from relax.optim.linmin import *

# objective function
def objective(vector, x) -> float:
	return vector[0]**x-vector[1]**x
 
# derivative of objective function
def gradient(vector, x):
    return np.array([
        x*vector[0]**(x-1),
        x*vector[1]**(x-1)
    ])

def hessian(vector, x):
    return np.array([
        [2*x*vector[0]**(x-2), 0],
        [0, 2*x*vector[1]**(x-2)]
    ])

if __name__=='__main__':
	
    vector = np.random.rand(2)
    print(vector)
    input()
	
    lnsearch = LnSearch(
		max_step=1,
		min_step=1e-5,
		schedule=100,
		exponent=0.999,
		order=10,
		gnorm=0
	)
    optimizer = CubicMin(lnsearch)
    rng = np.random.default_rng(0)
	
    for i in range(100):

        grad_np = gradient(vector=vector, x=2)
        hessian_np = hessian(vector=vector, x=2)
        print(hessian_np)

        optimizer.completion_check(np.linalg.norm(grad_np))
        vector = optimizer.step(
            grad=grad_np, params=vector, 
            line_search_fn='steady_step', 
            hessian=hessian_np,
            debug=True, rng=rng)
        print(vector)
        input()
