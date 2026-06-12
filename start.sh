#!/bin/bash
# start.sh — Launch the AI Router platform using uv

echo "=== AI Router Platform ==="
echo ""

# Check uv
if command -v uv &>/dev/null; then
    echo "[uv] Using uv for dependency management"
else
    echo "ERROR: uv is required. Install it: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Sync dependencies
echo "[1/3] Syncing dependencies..."
uv sync 2>/dev/null || uv pip install -e . 2>/dev/null || echo "  (sync skipped — running with existing env)"

# Install network-rl (optional, for model inference)
echo "[2/3] Checking network-rl package..."
if [ -d "模型项目/network-rl" ]; then
    uv pip install -e "模型项目/network-rl" 2>/dev/null || echo "  (network-rl install skipped — model inference requires torch deps)"
else
    echo "  (network-rl not found — model inference disabled, OSPF baseline available)"
fi

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "  [WARN] Docker not found — /api/deploy disabled, use /api/infer instead"
fi

# Start backend
echo "[3/3] Starting backend server..."
echo ""
echo "  Backend:  http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo "  Frontend: open topology-editor.html in browser"
echo "  Endpoints:"
echo "    POST /api/infer  — lightweight inference (no Docker needed)"
echo "    POST /api/deploy — full Mininet pipeline (requires Docker)"
echo "    POST /api/chat   — AI Agent chat (requires ANTHROPIC_API_KEY)"
echo ""
cd backend && uv run python run.py
