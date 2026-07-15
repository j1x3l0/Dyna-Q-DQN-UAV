#!/bin/bash

echo "=========================================="
echo "  UAV-Assisted Wireless Sensor Networks"
echo "  Deep Reinforcement Learning Training"
echo "=========================================="

echo ""
echo "Step 1: Check GPU status"
nvidia-smi
echo ""

echo "Step 2: Check Python environment"
python3 --version
echo ""

echo "Step 3: Install CUDA version PyTorch (GPU accelerated)"
echo "  - Check above for CUDA Version in nvidia-smi output"
echo "  - Choose the appropriate command below:"
echo ""
echo "  For CUDA 12.1 (recommended):"
echo "    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo ""
echo "  For CUDA 11.8:"
echo "    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo ""
echo "  Then install other dependencies:"
echo "    pip install numpy matplotlib"
echo ""

echo "Step 4: Create results and logs directories"
mkdir -p results logs
echo ""

echo "=========================================="
echo "  Available Training Scripts:"
echo "  ----------------------------"
echo "  1. train_all.py          - k sweep (0/1/3) x seeds (42/123/2026)"
echo "  2. train_maddpg.py       - Pure MADDPG"
echo "  3. train_hierarchical.py - MADDPG + Dyna-Q"
echo "  4. run_hierarchical_no_dyna.py - MADDPG + DQN (no Dyna)"
echo "=========================================="
echo ""

echo "=========================================="
echo "  RECOMMENDED: Run ALL THREE approaches"
echo "  (500 episodes each run + summary report)"
echo "=========================================="
echo ""
echo "Run in foreground:"
echo "  python3 scripts/train_all.py"
echo ""
echo "Run in background (recommended):"
echo "  nohup python3 scripts/train_all.py > train_all.log 2>&1 &"
echo ""
echo "Run individual approaches:"
echo "  nohup python3 scripts/train_maddpg.py > maddpg.log 2>&1 &"
echo "  nohup python3 scripts/train_hierarchical.py > hierarchical.log 2>&1 &"
echo "  nohup python3 scripts/run_hierarchical_no_dyna.py > hierarchical_no_dyna.log 2>&1 &"
echo ""

echo "To check running processes:"
echo "ps aux | grep python"
echo ""

echo "To check GPU usage during training:"
echo "watch -n 2 nvidia-smi"
