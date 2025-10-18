#!/bin/bash

echo "🔧 Fixing Dependencies for Python 3.11+ Compatibility"
echo "=================================================="

# Remove any existing pickle5 installation that might be causing conflicts
echo "📦 Removing potentially problematic packages..."
pip uninstall pickle5 -y 2>/dev/null || echo "   pickle5 not installed"

# Install core dependencies first
echo "📦 Installing core dependencies..."
pip install "torch>=2.0.0" "torchvision>=0.15.0" "numpy>=1.21.0"

# Install DEAP for evolutionary algorithms
echo "📦 Installing DEAP (evolutionary algorithms)..."
pip install "deap>=1.3.0"

# Install visualization libraries
echo "📦 Installing visualization libraries..."
pip install "matplotlib>=3.5.0" "seaborn>=0.11.0"

# Install other essentials
echo "📦 Installing other dependencies..."
pip install "pandas>=1.3.0" "scikit-learn>=1.0.0" "pyyaml>=6.0" "tqdm>=4.60.0"

# Install Jupyter and notebook support
echo "📦 Installing Jupyter notebook support..."
pip install "jupyter>=1.0.0" "ipykernel>=6.0.0" "ipywidgets>=7.6.0"

# Install data storage (excluding problematic pickle5)
echo "📦 Installing data storage libraries..."
pip install "h5py>=3.6.0"

# Optional packages
echo "📦 Installing optional packages..."
pip install "joblib>=1.1.0" "plotly>=5.0.0"

echo ""
echo "✅ Dependencies installed successfully!"
echo "🐍 Python $(python --version) with built-in pickle protocol 5"
echo "🎯 Ready to run the hyperparameter optimization experiment!"
echo ""
echo "Next steps:"
echo "1. Run the Jupyter notebook"
echo "2. Or use: python run_experiment.py --mode light"