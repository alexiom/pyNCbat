# =========================================================================== #
#                                 SCRIPT SETUP                                #
# =========================================================================== #

# script parameters
verbSolve = True    # print solver iterations?

# import python libraries              # initialize timer for script
import os
from time import time as script_timer; start_time = script_timer()

# import code written for NC battery project
from helpers.functions import find_subdir, RunTime_Message, Show_Case
from parameters.storage import ALPHASL, DELTASL
from parameters.solver import MAX_ITER, TOL, TOL_INF, TOL_GRAD
from NCbat_v3 import ElectricityModel, ModelCase, ConfigureStorage, ModelSolver, BatterySolution

# Get parameters from shell environment
CASEID = os.getenv('SLURM_ARRAY_TASK_ID')
RUN    = os.getenv('RUN')

# Get project directory
PROJDIR = find_subdir('data')

# Read In Common Data for Electricity Problem
ElectricityModel.ImportData(PROJDIR+'data/')

print('Solving NC Battery Model For Case '+CASEID+'and Run'+RUN+'!\n')

# ============================================================================ #
#                    SOLVE MODEL WITH ONE STORAGE TECHNOLOGY                   #
# ============================================================================ #

# Initialize ModelCase Instance for One Storage Technology
Case2 = ModelCase(2,caseID=CASEID,param_file='scenario_grid.csv')  # initialize problem
print('')
Show_Case(Case2.case,Nw=0) # show case parameters

# Configure Parameters of Sinlge Storage Technology
Storage2 = ConfigureStorage(ALPHASL, DELTASL, Case2)

# Initialize Solver
#equ_cons=range(Nh*Ngen,Nh*Ngen+Nh)
Solver = ModelSolver(tol = TOL, tol_inf = TOL_INF, tol_grad = TOL_GRAD)

# initialize solver for one storage technology (alpha can be less than 1)
OSQPsolver2 = Solver.setup(Case2, Storage2, verbose = verbSolve,
                            max_iter = MAX_ITER, alpha = 1.6,
                            adaptive_rho_tolerance = 4,
                            adaptive_rho_interval = 25)
    
# solve model with one storage technology
OSQPsol2 = Solver.solve(OSQPsolver2)

# =========================================================================== #
#                        Format and Export Solution                           #
# =========================================================================== #

BatSol = BatterySolution(Case2, Storage2, Solver, OSQPsol2)

BatSol.export_hourly_data(fname='hourly_ID'+CASEID+'.csv',out_path=PROJDIR+'output/simulated_output/')
BatSol.export_solution_summary(fname='output_ID'+CASEID+'.csv',out_path=PROJDIR+'output/simulated_output/')

rt = script_timer() - start_time # 
print('Done!', RunTime_Message(rt), f'Iterations = {BatSol.iter:,}\n\n')
