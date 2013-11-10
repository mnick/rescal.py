# coding: utf-8
# Copyright (C) 2013 Maximilian Nickel <mnick@mit.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
RESCAL Python package

This package provides routines to compute various forms of
the RESCAL tensor factorization.

RESCAL factors a (usually sparse) three-way tensor $\ten{X}$ such that each
frontal slice $X_k$ is factored into

.. math:: X_k = A * R_k * A.T.

The frontal slices of a tensor are $N \times N$ matrices. Usually, these
matrices correspond to the sparse adjacency matrices of the relational graph
for a particular relation in a multi-relational data set.

See
---
For a full description of the algorithm see:
.. [1] Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
       "A Three-Way Model for Collective Learning on Multi-Relational Data",
       ICML 2011, Bellevue, WA, USA

.. [2] Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
       "Factorizing YAGO: Scalable Machine Learning for Linked Data"
       WWW 2012, Lyon, France
"""

from .rescal import als as rescal_als
