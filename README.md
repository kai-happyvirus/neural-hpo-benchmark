# Hyperparameter Optimization with Evolutionary Algorithms

## 🎯 Best Practices Implementation for Cross-Platform Deployment

This project implements industry-standard best practices for hyperparameter optimization, comparing evolutionary algorithms (GA, DE, PSO) with traditional methods. Designed for reliable academic evaluation across different systems and hardware configurations.

**✨ Best Practice Features:**
- 🔧 **Automatic system detection** and hardware optimization
- 🛡️ **Cross-platform safety** with multiprocessing compatibility
- ⚡ **Dynamic configuration** adapting to available resources
- 🎯 **Flexible algorithm selection** with simple command structure
- � **Essential-only codebase** removing unnecessary complexity

### 📋 System Requirements
- Python 3.8+ (automatically validated)
- Cross-platform: macOS, Windows, Linux, Google Colab
- Automatic hardware detection: CUDA GPU, Apple Silicon MPS, or CPU fallback
- Self-configuring multiprocessing with platform-specific safety measures

### 🚀 Installation & Setup (Best Practices)

**Option 1: Intelligent Auto-Setup (Recommended)**
```bash
python setup_system.py
```

**Option 2: Full System Analysis**
```bash
python setup_system.py --full
```

**Option 3: Test Current Configuration**
```bash
python setup_system.py --test
```

**Option 4: Manual Installation (if auto-setup unavailable)**
```bash
pip install -r requirements.txt
```

### 🎬 Running Experiments

## **📊 Essential Commands (Best Practice Workflow)**

**1️⃣ For Video Demo/Quick Test (2-5 minutes):**
```bash
python run_experiment.py light
```

**2️⃣ For Full Research Results:**
```bash
python run_experiment.py full
```

## **🎯 Optional: Specific Algorithm Testing**

**Run only specific algorithm:**
```bash
python run_experiment.py full --model ga          # Only Genetic Algorithm
python run_experiment.py light --algorithm pso    # Only PSO in demo mode
```

**🔬 Algorithm Order (when running full):**
- **Traditional first**: `grid` → `random` (baselines)  
- **Evolutionary next**: `ga` → `de` → `pso` (research comparison)

## **📓 Interactive Jupyter Notebook:**
```bash
jupyter notebook Hyperparameter_Optimization_Evolutionary_Algorithms.ipynb
```

### 📊 What You'll Get

- **Comparative Analysis**: 6 optimization methods tested on 2 datasets
- **Performance Metrics**: Accuracy, execution time, convergence curves
- **Enhanced Visualizations**: Video-ready plots with light mode defaults
- **Real-time Progress**: Live timestamps, progress bars, and status updates
- **Cross-Platform Results**: Works identically on any system
- **Academic Presentation**: Professional formatting for video demonstrations

### 📁 Essential Files

**🚀 Main Commands:**
- `run_experiment.py` - **The only script you need to run experiments**
- `requirements.txt` - Python dependencies
- `simple_setup.sh` - One-click setup (optional)

**📊 Core System:**
- `config/config.yaml` - Configuration settings
- `src/` - All source code modules
- `Hyperparameter_Optimization_Evolutionary_Algorithms.ipynb` - Interactive notebook

### �🔧 Troubleshooting

**Issue: pickle5 build errors**
- Solution: Run `./simple_setup.sh` (automatically removes incompatible pickle5)

**Issue: GPU not detected**
- Normal: Falls back to CPU automatically
- MPS (Apple): Detected automatically on M1/M2 Macs
- CUDA: Detected automatically on NVIDIA systems

**Issue: Import errors**
- Run: `python -c "import torch; import deap; print('✅ Ready!')"`

### 📁 Simple Project Structure

```
├── run_experiment.py          # 🎯 MAIN SCRIPT - Run this!
├── requirements.txt            # Dependencies  
├── simple_setup.sh            # Quick setup (optional)
├── config/config.yaml         # Settings
├── src/                       # Source code
├── Hyperparameter_Optimization_Evolutionary_Algorithms.ipynb  # Notebook
├── results/                   # Generated results
└── figures/                   # Generated plots
```

### 🎓 Academic Features

- **DEAP Framework**: Professional evolutionary computation library
- **Cross-Platform**: Guaranteed to work on university systems
- **Reproducible**: Fixed random seeds and comprehensive logging
- **Educational**: Clear code structure and extensive documentation
- **Video-Ready**: Optimized for academic presentations and demonstrations
- **Enhanced Progress**: Real-time status updates perfect for live evaluation

### 🎥 Video Demonstration Features

- **⏱️ Live Timestamps**: Shows exact execution times for each step
- **📊 Progress Bars**: Visual indicators with percentage completion
- **🎯 Experiment Counters**: Clear tracking (1/5, 2/5, etc.)
- **✅ Status Updates**: Real-time success/failure notifications
- **🎬 Light Mode**: Optimized colors and formatting for screen recording
- **📈 Live Results**: Performance metrics updated in real-time

---

**Ready for evaluation and video demonstration!** 🎯

*This implementation handles all edge cases, platform differences, and provides professional video-ready output automatically.*