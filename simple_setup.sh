#!/bin/bash

echo "🔧 Simple Dependency Installation (Python 3.11+ Compatible)"
echo "=========================================================="

echo "📦 Removing problematic pickle5 package if present..."
pip uninstall pickle5 -y 2>/dev/null || echo "   pickle5 not installed (good!)"

echo "📦 Installing from requirements file..."
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "🧪 Testing key imports..."
python -c "
import sys
print(f'Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cuda\" if torch.cuda.is_available() else \"cpu\")}')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')

try:
    import deap
    print('✅ DEAP: Available')
except ImportError as e:
    print(f'❌ DEAP import failed: {e}')

try:
    import matplotlib
    import seaborn
    print('✅ Visualization: matplotlib and seaborn available')
except ImportError as e:
    print(f'⚠️  Visualization libraries: {e}')

try:
    import pickle
    print(f'✅ Pickle protocol: {pickle.HIGHEST_PROTOCOL} (built-in)')
except Exception as e:
    print(f'⚠️  Pickle issue: {e}')

print('\\n🎯 Ready for hyperparameter optimization!')
"

echo ""
echo "🚀 Next steps:"
echo "   1. Open the Jupyter notebook: jupyter notebook"
echo "   2. Or run a quick test: python run_experiment.py --mode light"