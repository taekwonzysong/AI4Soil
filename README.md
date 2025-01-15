## 1. ABOUT THE REPOSITORIES

Title: Repositories for A Novel Numerical Method for Agro-Hydrological Modeling of Water Infiltration in Soil

Creators: Zeyuan Song and Zheyu Jiang

Organization: School of Chemical Engineering, Oklahoma State University

Description: This repository contains the Python codes and Jupyter Notebook implementation for the 1-D and 3-D case studies presented in our manuscript "A Novel Data-driven Numerical Method for Hydrological Modeling of Water Infiltration in Porous Media", which has been submitted to Chemical Engineering Science for publication. 

Related publication: Song, Z., & Jiang, Z. (2023). A Novel Data-driven Numerical Method for Hydrological Modeling of Water Infiltration in Porous Media. arXiv preprint arXiv:2310.02806.

## 2. TERMS OF USE

This repository is licensed under [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## 3. CONTENTS

File listing 

- main.py: main file for solving the case study problem
- matrix.py: functions to calculate the condition number and spetral radius of the coefficient matrix A
- models.py: defines the neural network architectures
- training.py: function to train the neural networks
- data_processing.py: function to load the data
- GRW_solver.ipynb: a visualization of GRW results presented in Figure 5 of our manuscript
- DRW_solver.ipynb: a visualization of DRW results presented in Figure 5 of our manuscript

## 4. REFERENCES & ACKNOWLEDGEMENTS

Our DRW idea was inspired by the Global Random Walk concept proposed in:
1. Suciu, N., Illiano, D., Prechtel, A., Radu, F. A., 2021. Global random walk solvers for fully coupled flow and transport in saturated/unsaturated porous media, Advances in Water Resources 152 (2021) 103935.
2. Suciu, N., 2019. Diffusion in Random Fields: Applications to Transport in Groundwater, Springer.

Credit is also given to the following GitHub repository:
2. Suciu, N., Illiano, D., Prechtel, A., Radu, F. A., 2021. https://github.com/PMFlow/RichardsEquation

## 5. HOW TO CITE US

@misc{song2024noveldatadrivennumericalmethod,
      title={A Novel Data-driven Numerical Method for Hydrological Modeling of Water Infiltration in Porous Media}, 
      author={Zeyuan Song and Zheyu Jiang},
      year={2024},
      eprint={2310.02806},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2310.02806}, 
}
