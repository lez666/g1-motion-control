# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/hssim
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_isaacsim

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    if [[ ! -f $CONDA_ROOT/bin/mamba ]]; then
      $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    fi
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n hssim python=3.11 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate hssim

  # Install ffmpeg for video encoding
  conda install -c conda-forge -y ffmpeg
  conda install -c conda-forge -y libiconv
  conda install -c conda-forge -y libglu

  # Below follows https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
  # Install IsaacSim
  pip install --upgrade pip
  pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

  # Install dependencies from PyPI first
  pip install pyperclip
  # Install dependencies from regular PyPI (required by isaacsim-kernel but not in NVIDIA index)
  # Install base dependencies first (with specific versions required by isaacsim-kernel)
  pip install "attrs==25.1.0"
  pip install "multidict==6.1.0"
  pip install "yarl==1.18.3"
  pip install "idna==3.10"
  pip install "idna-ssl==1.1.0"
  pip install "typing-extensions==4.12.2" --force-reinstall
  pip install "async_timeout==5.0.1"
  # Now install aiohttp (will use the versions above)
  pip install "aiohttp==3.11.11"
  pip install "pycares==4.8.0"
  pip install "aiodns==3.1.1"
  pip install "Pillow==11.3.0"
  pip install "toml==0.10.2"
  pip install "click==8.1.7"
  pip install "websockets==12.0"
  pip install "fastapi==0.115.7"
  # Install numpy==1.26.0 (required by isaacsim-kernel, downgrade from torchvision's numpy 2.3.5)
  pip install "numpy==1.26.0" --force-reinstall
  # Then install isaacsim from NVIDIA index (use extra-index-url to allow fallback to PyPI for dependencies)
  pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com --trusted-host pypi.nvidia.com

  if [[ ! -d $WORKSPACE_DIR/IsaacLab ]]; then
    git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.0 $WORKSPACE_DIR/IsaacLab
  fi

  sudo apt install -y cmake build-essential
  cd $WORKSPACE_DIR/IsaacLab
  # work-around for egl_probe cmake max version issue
  export CMAKE_POLICY_VERSION_MINIMUM=3.5
  ./isaaclab.sh --install

 # Install Holosoma
  pip install -U pip
  pip install -e $ROOT_DIR/src/holosoma[unitree,booster]

  # Force upgrade wandb to override rl-games constraint
  pip install --upgrade 'wandb>=0.21.1'
  touch $SENTINEL_FILE
fi
