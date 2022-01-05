# efficientMPC
Efficient Model Predictive Control Implementation

The original algorithm can be found: https://www.researchgate.net/profile/Peter-Gibbens-2/publication/270872533_Efficient_Model_Predictive_Control_Algorithm_for_Aircraft/links/54bd90390cf27c8f2814bad5/Efficient-Model-Predictive-Control-Algorithm-for-Aircraft.pdf

You can change the system provided with your own and test for convergence of the Quadratic Optimization solver as well, using two QP algorithms

Open main.m

Define your system parameters and continous time model (LTI)

Define your operating point (states X at t = 0), and discretization time.

Define weight matrices, upper and lower input constraints (it has to be in order like the example), along with incremental inputs, leave dU blank if there are none

Update the constraints, specify the prediction horizon for each output (Np) - the example contains 2 unique ones.

Configure the controller and run 2 quadratic optimization iterations to ensure that it's feasible.



