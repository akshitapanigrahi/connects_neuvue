# connects_neuvue
a package with API defined for querying connects data products for neuvue visualizations and other downstream tasks


## 📦 Installation

### 🔹 Basic Installation

To install the package and its core dependencies, run:

```bash
pip install .
```

### 🔹 With Optional Dependencies

To include optional tools like `neuron_morphology_tools` from GitHub, use:

```bash
pip install .[morph_tools]
```

> 💡 `morph_tools` corresponds to the optional extras defined in `setup.py`. This package will help with any visualizations of the neuron networkx objects

### 🔹 Development Mode

If you're actively developing the package and want changes to reflect immediately:

```bash
pip install -e .
```

To include the optional extras in development mode:

```bash
pip install -e .[morph_tools]
```

### 🔹 Install Directly from GitHub (Optional)

If the project is hosted on GitHub, you can install it directly with:

```bash
pip install git+https://github.com/yourusername/your_project.git
```

## 📁 Project Structure

```
your_project/
├── your_project/           # Main package
│   ├── __init__.py
│   └── ...                 # Your modules
├── tests/                  # Unit tests
│   └── __init__.py
├── setup.py                # Installation script
├── requirements.txt        # Core dependencies
├── README.md               # Project overview
├── LICENSE                 # Optional
└── .gitignore              # Git ignore rules
```

