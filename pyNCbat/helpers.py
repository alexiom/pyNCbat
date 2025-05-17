
from pathlib import Path
import os.path, os
import numpy as np
from scipy import sparse as sp

# ============================================================================ #
#                  DEFINE UTILITY FUNCTIONS FOR PROJECT                        #
# ============================================================================ #

def _Sparse0(N,K):
    return sp.csc_matrix((N,K))
    
def _SparseI(N):
    return sp.identity(N,format="csc")

def find_subdir(target: str, start:Path = Path.cwd()) -> str:
    """
    Starting from `start`, look at each parent directory in turn.
    As soon as you find one that contains a subdirectory named `target`,
    return the relative path from `start` to that parent (e.g. '../../').
    
    Raises FileNotFoundError if no ancestor contains `target`.
    """
    start = start.resolve()
    # iterate over immediate parent first, then its parent, etc.
    for ancestor in start.parents:
        if (ancestor / target).is_dir():
            # compute relative path from start → ancestor
            rel = os.path.relpath(ancestor, start)
            # os.path.relpath gives e.g. '..' or '../..'
            # ensure it ends with a slash
            if not rel.endswith(os.sep):
                rel += os.sep
            return rel
    raise FileNotFoundError(f"No parent of {start} contains a subdir '{target}'")


# function for creating a message of the run time in 
#         years, months, days, hours, minutes, and seconds
def RunTime_Message(rt:float,show:bool=False):

    divs   = np.cumprod(np.array([1,60,60,24,30.41667,12]))[::-1]
    units  = ['second','minute','hour','day','month','year']
    units  = np.array(units)[::-1]
    mess   = 'Run Time = '

    for i, div_i in enumerate(divs):
        
        # get quotient and reminder from rt / div_i
        q, r = divmod(rt,div_i) 

        # Recursively Create Run Time Message
        # --- div_i must go into rt at least once to trim units with 0s
        if q > 0 or units[i]=='second':
            rt = r # update remaining run time
            if not units[i] == 'second':
                mess += (f"{q:.0f}" +' '+ units[i]) # update run time message 
            else:
                mess += (f"{q+r:.2f}" +' '+ units[i]) # update run time message 
            # if multiple, make units plural
            if q > 1:
                mess += 's'
            # if not last, add comma to seperate units
            if not units[i] == 'second':
                mess += ', '
            else:
                mess += "!"

    if show:
        print(mess)
    
    return mess


# function for displaying scenario parameters
def Show_Case(caseDF,Nw:int=0) -> None:

    # Create "Case n" string stem
    ID       = caseDF['caseID']  # store case index
    caseStem = f"Case {ID:,}:" # create case stem
    
    # Store Values of Parameters to Display 
    #  - (Stored In Same Order of Message)
    h0   = caseDF['batstarthour'] # starting hour of the battery
    rinv = caseDF['suninv']       # renewables capital cost innov (% as decimal) 
    binv = caseDF['batinv']       # battery capital cost innov (% as decimal) 
    Linv = caseDF['longinv']      # long duration capital cost innov (% as decimal) 
    dinc = caseDF['demandinc']    # data center demand increase (MWh)
    ctax = caseDF['ctax']         # size of carbon tax in dollars

    # Create Initial Strings
    h0Str     = f" Starting Hour = {h0:,}"
    renewStr  = f"Renew Cap Cost Drop = {100*rinv:.0f}% "
    batStr    = f"S/L Storage Cap Cost Drop = {100*binv:.0f}% / {100*Linv:.0f}%"
    demandStr = f"Demand Increase = {dinc:.0f} MWh "
    ctaxStr   = f"Carbonx Tax = ${ctax:.2f}"
    
    # Add Spacing Between First and Second Line Elements To Align
    Ndiff = len(demandStr)-len(renewStr) # number of char differece
    renewStr  += " "*(max(0, Ndiff)+1)   # add white space to line 1
    demandStr += " "*(max(0,-Ndiff)+1)   # add white space to line 2

    # Format Numbers and Create Message
    message  = " " * Nw  + caseStem     # "Case n" part of string
    message += " " * (Nw+10) + h0Str + "\n" # Add Bat Starting Hour String
    message += " "*(len(caseStem)+Nw+1) # Indent New Line
    message += renewStr                # Add Renew Innov Me String
    message += batStr           + "\n" # Add Battery Innov String
    message += " "*(len(caseStem)+Nw+1) # Indent New Line
    message += demandStr               # Add Demand Inc String
    message += ctaxStr                 # Add Carbon Tax String

    # Print Formatted Case String
    print(message)