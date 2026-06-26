# Physics-Informed Neural Networks (PINNs)

Forward **Physics-Informed Neural Network** projects for classic PDEs — Burgers', heat/diffusion, and Navier–Stokes flow (including particle tracing). Implemented in **PyTorch** (from scratch) and **[DeepXDE](https://deepxde.readthedocs.io/)**.

A PINN trains a neural network to satisfy a PDE by adding the PDE residual, boundary, and initial conditions directly into the loss function — so the governing physics is enforced during training instead of being learned purely from data.

---

## Projects

| Notebook | Problem | Framework |
|---|---|---|
| [`Burgers_1D_main.ipynb`](Burgers_1D_main.ipynb) | 1D viscous Burgers' equation | PyTorch |
| [`burgers_2D_main.ipynb`](burgers_2D_main.ipynb) | 2D Burgers' equation | NumPy / Matplotlib |
| [`heat_main.ipynb`](heat_main.ipynb) | 1D heat (diffusion) equation | NumPy / PyTorch |
| [`deepxde_1d_heat_main-2.ipynb`](deepxde_1d_heat_main-2.ipynb) | 1D heat equation | DeepXDE |
| [`Heat_2D_pinns_main.ipynb`](Heat_2D_pinns_main.ipynb) | 2D heat equation | PyTorch |
| [`ns_deepxde_main.ipynb`](ns_deepxde_main.ipynb) | 2D Navier–Stokes flow | DeepXDE |
| [`NS_Particle_Tracing_DeepXDE_main.ipynb`](NS_Particle_Tracing_DeepXDE_main.ipynb) | Navier–Stokes + particle tracing | DeepXDE |
| [`NS_Particle_Tracing_DeepXDE_Y_Shape_main.ipynb`](NS_Particle_Tracing_DeepXDE_Y_Shape_main.ipynb) | NS + particle tracing, Y-shaped channel | DeepXDE |
| [`NS_Particle_Tracing_DeepXDE_Y_Shape_main_ver2.ipynb`](NS_Particle_Tracing_DeepXDE_Y_Shape_main_ver2.ipynb) | NS + particle tracing, Y-shaped channel (v2) | DeepXDE |
| [`NS_particle.ipynb`](NS_particle.ipynb) | Particle-tracing helper / snippet | DeepXDE |
| [`Erfan_PyTorch4.ipynb`](Erfan_PyTorch4.ipynb) | PyTorch & autograd warm-up | PyTorch |

> `*.dat` files (e.g. `loss.dat`, `train.dat`, `test.dat`) are training/evaluation logs produced by the DeepXDE notebooks.

## Tech stack

- Python 3.9+
- [PyTorch](https://pytorch.org/), [DeepXDE](https://deepxde.readthedocs.io/)
- NumPy, SciPy, Matplotlib

## Getting started

```bash
git clone https://github.com/erfant00001/physics-informed-neural-networks.git
cd physics-informed-neural-networks
pip install torch deepxde numpy scipy matplotlib jupyter
jupyter notebook
```

Open any notebook and run the cells top to bottom.

## Acknowledgements

These projects were created while learning Physics-Informed Neural Networks and Scientific Machine Learning from the Udemy courses of **Dr. Mohammad Samara** (Data Science / Machine Learning expert; PhD, University of Tokyo).

Instructor profile: **https://www.udemy.com/user/mohammad-samara-18/**

The problem setups and course material are credited to the instructor. This repository contains my own implementations and notes produced while following the courses, shared for learning and reference.
