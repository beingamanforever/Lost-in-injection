# Source from the repository root before any command. Caches keep their usual defaults; an untracked
# env.local.sh can move them (on the A10Gs it points HF_HOME, uv and TMPDIR at the NVMe disk).
if [ -f env.local.sh ]; then source env.local.sh; fi
export NLTK_DATA=${NLTK_DATA:-$PWD/.cache/nltk-data}
# The Deep Learning AMI's LD_LIBRARY_PATH holds its own CUDA libraries (cuDNN 9.10 on the first A10G),
# which shadow the ones bundled with torch and break training; use the bundled ones everywhere.
unset LD_LIBRARY_PATH
# Driver 570 supports CUDA up to 12.8; take torch wheels built for it.
export UV_TORCH_BACKEND=cu128
