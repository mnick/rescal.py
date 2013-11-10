RESCAL
======

This module provides standalone routines to compute various forms of
the RESCAL tensor factorization.

RESCAL factors a (usually sparse) three-way tensor X such that each
frontal slice X_k is factored into

	X_k = A * R_k * A.T

The frontal slices of X are quadratic, possibly asymmetric N x N matrices. 
Usually, these matrices correspond to the sparse adjacency matrices of the 
relational graph for a particular relation in a multi-relational data set.

A copy of this standalone module of RESCAL is also included in the [scikit-tensor](https://github.com/mnick/scikit-tensor) libray.


Dependencies
------------
The required dependencies to build the software are `Numpy >= 1.3`, `SciPy >= 0.7`.


Usage
-----
Example script to decompose kinships data using RESCAL-ALS:

```python
import logging
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
from rescal import rescal_als

# Set logging to INFO to see RESCAL information
logging.basicConfig(level=logging.INFO)

# Load Matlab data and convert it to dense tensor format
T = loadmat('data/alyawarra.mat')['Rs']
X = [lil_matrix(T[:, :, k]) for k in range(T.shape[2])]

# Decompose tensor using RESCAL-ALS
A, R, fit, itr, exectimes = rescal_als(X, 100, init='nvecs', lambda_A=10, lambda_R=10)
```

For more examples on the usage of RESCAL, please see the [examples](examples) directory in the source tree.


References
----------
If you use `rescal.py` in your research, please cite

	Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
	"A Three-Way Model for Collective Learning on Multi-Relational Data",
	Proceedings of the 28th International Conference on Machine Learning (ICML'11), 
	809--816, ACM, Bellevue, WA, USA, 2011

or

	Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
    "Factorizing YAGO: Scalable Machine Learning for Linked Data"
	Proceedings of the 21st International Conference on World Wide Web (WWW'12),
	271--280, ACM, Lyon, France, 2012


Install
-------
This package uses distutils, which is the default way of installing python modules. To install in your home directory, use::

    python setup.py install --user

To install for all users on Unix/Linux

    python setup.py build
    sudo python setup.py install

To install in development mode

    python setup.py develop


Authors
-------
Maximilian Nickel <mnick AT mit DOT edu>

+ <http://twitter.com/mnick>
+ <http://github.com/mnick>


License
-------
rescal.py is licensed under the GPLv3 <http://www.gnu.org/licenses/gpl-3.0.txt>
