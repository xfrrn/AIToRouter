#!/bin/bash
# start.sh — Launch the AI Router platform

echo "=== AI Router Platform ==="
echo ""

# Check prerequisites
if ! command -v python &>/dev/null; then
    echo "ERROR: Python 3.12+ required"
    exit 1
fi

if ! command -v docker &>/dev/null; then
    echo "WARNING: Docker not found. Mininet deployment will not work."
    echo "         Install Docker to enable full pipeline."
fi

# Install backend dependencies
echo "[1/3] Installing backend dependencies..."
cd backend
pip install -r requirements.txt -q 2>/dev/null
cd ..

# Install network-rl
echo "[2/3] Installing network-rl package..."
cd "模型项目/network-rl"
pip install -e . -q 2>/dev/null || echo "  (network-rl editable install skipped — deps may need manual install)"
cd ../..

# Start backend
echo "[3/3] Starting backend server..."
echo ""
echo "  Backend:  http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo "  Frontend: open topology-editor.html in browser"
echo ""
cd backend && python run.py
