cat > ~/setup_hls_env.sh <<'EOF'
#!/usr/bin/env bash
# Load LCG view + activate venv for hls4ml/conifer on lxplus

# Stop if something fails
set -e

# ---- Choose your LCG view (must provide python >= 3.10) ----
LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_106a/x86_64-el9-gcc11-opt/setup.sh"

# ---- Your venv ----
VENV_DIR="$HOME/venvs/hls4ml_conifer"

# Load LCG
if [ -f "$LCG_SETUP" ]; then
  source "$LCG_SETUP"
else
  echo "ERROR: LCG setup not found at: $LCG_SETUP" >&2
  return 1 2>/dev/null || exit 1
fi

# Activate venv
if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
else
  echo "ERROR: venv not found at: $VENV_DIR" >&2
  echo "Create it with: python3 -m venv --system-site-packages $VENV_DIR" >&2
  return 1 2>/dev/null || exit 1
fi

# Quick sanity print (non-fatal)
echo "[hls-env] python: $(which python)"
python --version || true
EOF

chmod +x ~/setup_hls_env.sh
