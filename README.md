# g1-motion-control

## Overview

This repository provides an engineering implementation for motion control and imitation learning on the Unitree G1 humanoid robot. The project builds upon the holosoma framework (included as a submodule) and extends it with G1-specific configurations, training pipelines, and deployment workflows. The core objectives are to enable keyboard/gamepad-controlled crawling behaviors with adjustable forward speed and turning rate, as well as full-body motion tracking that can reproduce human motions and dance sequences through action retargeting.

## Core Features

- Command-conditioned crawling locomotion with adjustable forward velocity and turning rate
- Full-body motion tracking and retargeting for human motions and dance
- Integration with holosoma framework for training and simulation
- G1-specific robot models and configurations
- Keyboard and gamepad input interfaces for real-time control
- Motion retargeting pipeline for converting human motion capture data to G1 joint trajectories

## Repository Structure

```
g1-motion-control/
├── configs/              # G1-specific training and deployment configurations
├── docs/                 # Architecture documentation and roadmap
├── scripts/              # Utility scripts for setup and deployment
│   └── bootstrap.sh     # Initial submodule sync and setup
├── third_party/         # External dependencies
│   └── holosoma/        # Holosoma framework (submodule)
└── README.md            # This file
```

The `third_party/holosoma` directory contains the holosoma framework with its full codebase, including simulation environments, training pipelines, and retargeting tools. This repository serves as a thin wrapper that provides G1-specific extensions and configurations rather than a fork of holosoma.

## Getting Started

Clone the repository and initialize submodules:

```bash
git clone <repository-url>
cd g1-motion-control
git submodule update --init --recursive
```

Alternatively, use the provided bootstrap script:

```bash
./scripts/bootstrap.sh
```

Next steps depend on your use case:

- **Simulation setup**: See `third_party/holosoma/scripts/` for simulator-specific installation instructions (Isaac Sim, Isaac Gym, MuJoCo)
- **Training**: Configuration files and training scripts will be located in `configs/` (see holosoma documentation for training workflow)
- **Deployment**: See `third_party/holosoma/src/holosoma_inference/` for on-robot inference and control interfaces
- **Motion retargeting**: See `third_party/holosoma/src/holosoma_retargeting/` for motion capture data processing and retargeting tools
