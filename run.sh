#!/bin/bash

# Hyperparameter Optimization Experiment Runner
# Convenience scripts for different execution modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
}

# Check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        print_warning "Virtual environment not found. Creating one..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
}

# Activate virtual environment
activate_venv() {
    source venv/bin/activate
    print_status "Virtual environment activated"
}

# Install dependencies
install_deps() {
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Setup function
setup() {
    print_status "Setting up experiment environment..."
    check_python
    check_venv
    activate_venv
    install_deps
    print_success "Setup completed!"
}

# Run full experiment
run_full() {
    print_status "Starting FULL experiment (this may take several hours)..."
    print_warning "This will run all algorithms on all datasets with multiple runs"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        activate_venv
        python run_experiment.py full --name "full_experiment_$(date +%Y%m%d_%H%M%S)"
        print_success "Full experiment completed!"
    else
        print_status "Full experiment cancelled"
    fi
}

# Run light experiment for video demonstration
run_light() {
    print_status "Starting LIGHT experiment for demonstration..."
    print_status "This should complete in a few minutes"
    activate_venv
    python run_experiment.py light --name "light_demo_$(date +%Y%m%d_%H%M%S)"
    print_success "Light experiment completed!"
}

# Run specific algorithm
run_specific() {
    local algorithm=$1
    local dataset=${2:-mnist}
    local runs=${3:-1}
    
    if [ -z "$algorithm" ]; then
        print_error "Algorithm not specified"
        echo "Usage: $0 specific <algorithm> [dataset] [runs]"
        echo "Algorithms: ga, de, pso, grid, random"
        echo "Datasets: mnist, cifar10"
        exit 1
    fi
    
    print_status "Running $algorithm on $dataset ($runs runs)..."
    activate_venv
    python run_experiment.py specific --algorithm "$algorithm" --dataset "$dataset" --runs "$runs" --name "${algorithm}_${dataset}_$(date +%Y%m%d_%H%M%S)"
    print_success "Specific experiment completed!"
}

# Test installation
test_install() {
    print_status "Testing installation..."
    activate_venv
    
    # Test imports
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')

import deap
print(f'DEAP version: {deap.__version__}')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('All imports successful!')
"
    
    print_success "Installation test completed!"
}

# Show help
show_help() {
    echo "Hyperparameter Optimization Experiment Runner"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  setup                    - Set up environment and install dependencies"
    echo "  full                     - Run full experiment (all algorithms, all datasets)"
    echo "  light                    - Run light experiment for demonstration"
    echo "  specific <alg> [ds] [n]  - Run specific algorithm"
    echo "    <alg>                  - Algorithm: ga, de, pso, grid, random"
    echo "    [ds]                   - Dataset: mnist (default), cifar10"
    echo "    [n]                    - Number of runs (default: 1)"
    echo "  test                     - Test installation"
    echo "  help                     - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup                 - Set up environment"
    echo "  $0 light                 - Quick demo"
    echo "  $0 specific ga mnist 3   - Run GA on MNIST 3 times"
    echo "  $0 full                  - Run complete experiment"
}

# Check if results directory exists and show recent results
show_results() {
    if [ -d "results" ]; then
        print_status "Recent experiment results:"
        ls -la results/ | tail -5
        echo ""
        print_status "To view results, check the 'results' directory"
        print_status "Each experiment folder contains:"
        echo "  - figures/: Generated plots and visualizations"
        echo "  - results/: Raw results data"
        echo "  - checkpoints/: Saved experiment state"
        echo "  - all_results.csv: Summary of all results"
    else
        print_warning "No results directory found. Run an experiment first."
    fi
}

# Main script logic
case "$1" in
    "setup")
        setup
        ;;
    "full")
        run_full
        ;;
    "light")
        run_light
        ;;
    "specific")
        run_specific "$2" "$3" "$4"
        ;;
    "test")
        test_install
        ;;
    "results")
        show_results
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    "")
        print_error "No command specified"
        show_help
        exit 1
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac