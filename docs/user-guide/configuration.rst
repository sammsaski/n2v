Configuration
=============

n2v provides a global configuration system for controlling parallelization and
LP solver settings.

Parallel LP Solving
-------------------

Many verification operations require solving multiple linear programs.
Parallelizing these LP solves can significantly speed up computation.

.. code-block:: python

   import n2v
   import multiprocessing

   # Enable parallel with all CPU cores
   n2v.set_parallel(True, n_workers=multiprocessing.cpu_count())

   # Auto-parallel: only parallelize for high-dimensional problems
   n2v.set_parallel('auto', threshold=10)

   # Disable parallel
   n2v.set_parallel(False)

LP Solver Selection
-------------------

n2v supports multiple LP solver backends:

.. code-block:: python

   import n2v

   # Use scipy linprog (HiGHS) -- recommended, ~1.5-2x faster than default
   n2v.set_lp_solver('linprog')

   # Use CVXPY with CLARABEL (default)
   n2v.set_lp_solver('default')

   # Use Gurobi (requires license)
   n2v.set_lp_solver('GUROBI')

See :doc:`lp-solvers` for a detailed comparison of LP solver performance.

Viewing Configuration
---------------------

.. code-block:: python

   import n2v

   # Get current config as dict
   config = n2v.get_config()
   print(config)

   # Access the global Config object directly
   print(n2v.config)
