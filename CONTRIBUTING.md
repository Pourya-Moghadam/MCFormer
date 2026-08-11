# Contributing

Contributions should preserve the manuscript contract and scientific provenance.

1. Create a focused branch and describe the scientific or engineering motivation.
2. Do not commit datasets, checkpoints, generated outputs, credentials, or private paths.
3. Add tests for behavior changes and update configuration/documentation when a public contract
   changes.
4. Run `make check` in the Python 3.11 environment.
5. Report deviations from the manuscript explicitly; never alter expected paper numbers in code.

Bug reports should include the resolved configuration, command, package versions, device details,
relevant hashes, and the smallest non-private traceback or reproduction possible.
