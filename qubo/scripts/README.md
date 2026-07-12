"""Scripts in this folder expect to be run from ``code-to-commit/``.

Canonical master is the **48-qubit** hourly QUBO in ``qubo/matrices/``.

Examples::

    python -m qubo.export_matrices
    python -m qubo.hybrid_decomposition
    python qubo/scripts/make_cim_spin_graph.py
    python qubo/scripts/make_qaoa_circuit.py
    python qubo/scripts/make_hybrid_figure.py

Figure helpers may still use relative paths; set cwd to ``code-to-commit``.
"""
