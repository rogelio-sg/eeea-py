# Installation

## Requirements

- Python 3.9 or higher
- NumPy ≥ 1.24

## Install from source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/rogelio-sg/eeea-py.git
cd eeea-py
pip install -e .
```

## Install dependencies

```bash
pip install -r requirements.txt
```

For development tools (testing, linting):

```bash
pip install -r requirements-dev.txt
```

## Verify the installation

```python
import eeea_py
print(eeea_py.__version__)
```

If no error is raised, the package is correctly installed.
