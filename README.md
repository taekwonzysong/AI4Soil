## 1. ABOUT THIS REPOSITORY

Title: A Novel Numerical Method for Agro-Hydrological Modeling of Water Infiltration in Soil

Creators: Zeyuan Song and Zheyu Jiang

Organization: School of Chemical Engineering, Oklahoma State University

Description: This repository contains the Python codes for 1-D through 3-D case studies discussed in the manuscript "A Novel Data-driven Numerical Method for Hydrological Modeling of Water Infiltration in Porous Media", which has been for publication. It also contains the datasets used to train the neural networks and Jupyter Notebook implementation for the 1-D case study.

## 2. TERMS OF USE

This repository is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## 3. CONTENTS

Under the 1-D Benchmark Problem folder:
- main.py: main file for solving the case study problem
- matrix.py: functions to calculate the condition number and spetral radius of the coefficient matrix A
- models.py: defines the neural network architectures
- training.py: function to train the neural networks
- data_processing.py: function to load the data
- GRW_solver.ipynb: a visualization of GRW results presented in Figure 5 of our manuscript
- DRW_solver.ipynb: a visualization of DRW results presented in Figure 5 of our manuscript
- Data/ground_truth_solutions.csv: "ground truth" solutions of Celia's problem obtained from finite difference method implemented in [SimPEG](https://simpegdocs.appspot.com/content/examples/20-published/plot_richards_celia1990.html)
- Data/reference_solutions_1-D_1.csv: reference solutions after performing data augmentation and will be used for neural network training
- Data/reference_solutions_1-D_2.csv: 
- Data/GRW_solutions_s1.csv: 

Under the 2-D Benchmark Problem folder:
- main_GRW.py: main file for running the GRW algorithm
- main_DRW.py: main file for running the DRW algorithm
- matrix.py: functions to calculate the condition number and spetral radius of the coefficient matrix A
- models.py: defines the neural network architectures
- training.py: function to train the neural networks
- data_processing.py: function to load the data
- Data/reference_solutions_2-D.csv: reference solutions after performing data augmentation and will be used for neural network training
- Data/reference_solutions_processed.csv: 

Under the 3-D Benchmark Problem folder:
- main_GRW.py: main file for running the GRW algorithm
- main_DRW.py: main file for running the DRW algorithm
- matrix.py: functions to calculate the condition number and spetral radius of the coefficient matrix A
- models.py: defines the neural network architectures
- training.py: function to train the neural networks
- data_processing.py: function to load the data
- Data/reference_solutions_2-D.csv: reference solutions after performing data augmentation and will be used for neural network training
- Data/reference_solutions_processed.csv: 

Under the Ref. [27] folder within the 3-D Benchmark Problem folder:
- GRW_Ref_27.m: runs the in-house GRW algorithm (this needs to be run first) for Ref. [27] in our manuscript
- DRW_Ref_27.m: performs data augmentation on the GRW solutions and runs the DRW algorithm for Ref. [27] in our manuscript
- theta_GM: function for calculating theta

## 4. ACKNOWLEDGEMENT

Our DRW framework is built upon the original GRW concept developed in:
1. Suciu, N., Illiano, D., Prechtel, A., Radu, F. A., 2021. Global random walk solvers for fully coupled flow and transport in saturated/unsaturated porous media, Advances in Water Resources 152 (2021) 103935.
2. Suciu, N., 2019. Diffusion in Random Fields: Applications to Transport in Groundwater, Springer.

We also acknowledge the [GitHub repository](https://github.com/PMFlow/RichardsEquation) where the code for Suciu et al. (2021) is presented.

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
