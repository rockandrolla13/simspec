# RFQ Alpha Strategy Simulator

## Skills & Reference
Read and follow all instructions in /home/ak/blueprint/ before starting any task.
These contain workflow skills and coding standards.

## Specification
Key files (in this directory):
- sim_spec.tex — primary simulator specification (LaTeX source)
- sim_spec.pdf — compiled specification document

Always consult the relevant spec section before implementing.

## Goal
Build a Jupyter notebook simulator in notebooks/ following sim_spec.tex exactly.

## Conventions
- Python 3.11+, numpy, scipy, pandas, matplotlib
- All parameters in a single dataclass with defaults from the spec
- Seed all randomness for reproducibility
- Each spec section → one notebook cell or module
