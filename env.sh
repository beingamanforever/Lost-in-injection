# Source on the A10G before any command: the root disk is full, so every cache lives on NVMe.
export PROJECT_ROOT=/opt/dlami/nvme/aman/lost-in-injection
export HF_HOME=/opt/dlami/nvme/aman/hf
export UV_CACHE_DIR=/opt/dlami/nvme/aman/uv-cache
export UV_PYTHON_INSTALL_DIR=/opt/dlami/nvme/aman/uv-python
export TMPDIR=/opt/dlami/nvme/aman/tmp
export PATH=/opt/dlami/nvme/aman/tools/bin:$PATH
mkdir -p "$TMPDIR"
