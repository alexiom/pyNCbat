import osqp
import pandas as pd
import numpy as np
from math import isclose
from numpy.typing import ArrayLike
from scipy import sparse as sp
from typing import Optional, Union

from helpers.functions import _Sparse0, _SparseI
from helpers.types import ArrayOrNum, OSQP_interface
from setup.scenarios import Scenarios

# =========================================================================== #
#                             Define Classes                                  #
# =========================================================================== #

class ElectricityModel:

    CO2GasCC = 0.74529/2.205
    CO2GasP  = 1.15889/2.205
    GWconv   = 1000
    
    # initialize ElectricityModel instance (NOT ACTUALLY NEEDED)
    def __init__(self) -> None:
        pass
    # end __init__ method

    # Import and Format Sample Data
    # Doing this as a class method and making the data a class variable
    # allow all instances of this class (and sub class) to access the data
    @classmethod
    def ImportData(cls,PATH) -> None:

        GWconv = cls.GWconv

        # if lasts character isn't /, add it to end
        if PATH[-1] != '/':
            PATH = PATH + '/'
        # end if PATH ends in / 

        # set transfer path and conversion factor as CLASS variables
        cls.PATH = PATH

        # read in data sample with variable names into DataFrame
        df = pd.read_csv(cls.PATH+'CAR_sample.csv') 

        df = df.rename(columns={'refdemand': 'demand', 'refprice': 'price'})
        
        # scale variables by conversion factor
        # --- variables to divide by conv
        dvars     = ['demand','ngnuc','ngwat']
        df[dvars] = df[dvars]/GWconv
        # --- variables to multiply by conv
        df['price'] = df['price']*GWconv

        # add df to class to make the data a class variable
        cls.DF = df
        
        # store numhours as class variable
        cls.numhours = df.shape[0]
    # end ImportData method
# end ElectricityModel class


# Define Battery Problem 
class ModelCase(ElectricityModel):

    # initialize ModelCase instance
    def __init__(self, Nstor: int, Scenarios:ScenarioGrid, caseID:Union[int,str] = 1, NstorMax:int = 2) -> None:

        if Nstor > NstorMax:
            raise Exception('Currently, Nstor must be either 1 or 2')
        # end if Nstor larger than NstorMax

        if isinstance(caseID,str):
            if not caseID.isdigit():
                raise ValueError(f"caseID expected an integer or digit only string, got {caseID!r}")
            else:
                caseID = int(caseID)
            # end if caseID is not a digit
        # end if caseID is an instance of a string

        self.NstorMax = NstorMax
        
        # Store Objects for Cleaner Code
        numhours = self.numhours

        #  Read In Data Parameter Data 
        case = Scenarios[caseID-1]

        # Set Storage Technologies
        if Nstor == 1:
            storVars = ('Bat')
        else:
            storVars = ('Bat','Long')

        # Get Technologies Used To Generate Eletricity
        # --- StorPrfx contains the prefixes used for indicating storage technology
        genCapVars  = [v for v in dfVars if v.startswith("cap") and (not v.endswith(StorPrfx))] # capital costs
        genMCVars   = [v for v in dfVars if v.startswith("mar") and (not v.endswith(StorPrfx))] # marginal costs
        genMC0Vars  = [v for v in genMCVars if isclose(dfPar[v],float(0))]                                      # renewable techs
        
        # Set number of technologies of all types
        Ngen   = len(genMCVars)  # number of generation techs
        Nrenew = len(genMC0Vars) # number of renewable generation techs
        Ntech  = Ngen + Nstor    # total number of techs

        #        Qt (numhours); qit (Ngen)*numhours); Ki (Ngen);      Sbar (1); St (numhours); b^d_t (numhours)
        #                                                                    | -------- Per Storage Tech ----------- |
        numvars = numhours    + Ngen*numhours       + Ngen +     Nstor*(  1    +   numhours   +   numhours)        

        # Get generation and renewable technologies 
        # --- strip cap/mar from capital and marginal cost variable names
        genVars   = [s.replace('cap','') for s in genCapVars]
        renewVars = [s.replace('mar','') for s in genMC0Vars]
        # --- convert first letter to lower case
        genVars   = [s[:1].lower() + s[1:] for s in genVars]
        renewVars = [s[:1].lower() + s[1:] for s in renewVars]
        # --- make sure nuclear is nuke
        genVars   = ['nuke' if s.startswith('nuk') else s for s in genVars]
        renewVars = ['nuke' if s.startswith('nuk') else s for s in renewVars]
        # --- store generation techs and renewable techs in class instance
        self.genVars = genVars; self.renewVars = renewVars

        # Get Capacity & Marginal Costs
        # --- initialize vectors to store costs
        CapCost = np.empty(Ntech)
        MarCost = np.empty(Ntech)
        # -- add capital and marginal costs of storage techs to capVars and mcVars
        capVars = genCapVars + ['cap'+v_i for v_i in storVars]
        mcVars  = genMCVars  + ['mar'+v_i for v_i in storVars]
        # --- Store Capital Costs
        for i, v_i in enumerate(capVars):
            CapCost[i] = float(dfPar[v_i])*self.GWconv
        # end for each technology capacity cost loop
        # --- Store Marginal Costs
        for i, v_i in enumerate(mcVars):
            MarCost[i] = float(dfPar[v_i])*self.GWconv
        # end for each technology marginal cost loop
        
        # Store Parameters of Class Instance i.e. parameters of the case
        cVars = ['caseID','suninv','windinv','nukinv','ctax','batinv','longinv','batstarthour','demandinc']
        
        # Objects straight from dfPar
        self.region     = dfPar['region']
        self.elasticity = dfPar['elasticity']
        self.suninv     = dfPar['suninv']
        self.windinv    = dfPar['windinv']
        self.nukinv     = dfPar['nukinv']
        self.carbontax  = dfPar['ctax']
        self.batinv     = dfPar['batinv']
        self.longinv    = dfPar['longinv']
        self.batstart   = int(dfPar['batstarthour'])
        self.demandinc  = dfPar['demandinc'] / self.GWconv
        self.case       = dfPar[cVars]

        # Calculate/Constructed Objects
        self.numhours = numhours; self.numvars  = numvars
        self.Ngen     = Ngen;     self.Nrenew   = Nrenew 
        self.Nstor    = Nstor;    self.Ntech    = Ntech
        self.CapCost  = CapCost;  self.MarCost  = MarCost 

        # get hourly IDs for shifting arrays to battery starting hour
        h0 = self.batstart-1; Nh = self.numhours
        self.hIDs = np.append(np.arange(h0,Nh),np.arange(0,h0))

        self.recover_demand()
        self.create_Pq()
        self.create_A2()
    # end __init__ method

    # Recover demand parameters from data
    #  - i.e. intercepts and slopes of linear demand
    def recover_demand(self, RemoveHydro:bool = True) -> None:
        
        DF = self.DF
        unitvec = np.ones(self.numhours)
        
        # Pull quantity demand and reference price
        refdemand = np.array(DF.demand)[self.hIDs]
        refprice  = np.array(DF.price)[self.hIDs]
        
        # Remove Hydro Demand if True
        if RemoveHydro:
            refdemand = np.subtract(refdemand,np.array(DF.ngwat)[self.hIDs])
        # end if to remove hydro

        # Recover Demand Curve Intercept and Slope
        self.DemA = refdemand * ((1-self.elasticity) * unitvec) + self.demandinc
        self.DemB = -1 * self.elasticity * unitvec * refdemand / refprice
    # end recover_demand method

    # Create Matrices That Change Over Cases
    def create_A2(self) -> None:

        # store variables for cleaner code
        # --- Counts/Numbers 
        Nh     = self.numhours
        Ngen   = self.Ngen
        Nrenew = self.Nrenew
        # --- cf varibales
        DF     = self.DF.iloc[self.hIDs]
        cffoss = np.ones((Ngen - Nrenew, Nh))
        cfmat  = np.vstack([DF['cfsun'],DF['cfwind'],cffoss]).T

        ############## CONSTRAINT 2
        # --- f_{it} K_i - q_{it} \geq 0, for all (i,t) \in 0:Ngen \times 0:Nh
        # Create fcons for each time period
        for t in range(0,Nh):
            temp = np.diag(cfmat[t,:])
            if t == 0:
                fcons = np.copy(temp)
            else:
                fcons = np.vstack([fcons,temp])
            # end if first time period or not
        # end for time period loop
        # Form A2
        self.A2 = sp.hstack([_Sparse0(Nh*Ngen,Nh), -1*_SparseI(Nh*Ngen),sp.csc_matrix(fcons), _Sparse0(Nh*Ngen,self.Nstor*(2*Nh+1))],format='csc')
    # end create_A2

    # Create Quadratic (Matrix) and Linear (Vector) Terms of Objective Function 
    def create_Pq(self) -> None:
        # Store variables for cleaner code
        Nh   = self.numhours; Nvars = self.numvars # num hours & variables
        Ngen = self.Ngen;     Nstor = self.Nstor   # num of MC gen techs & storage techs

        # Construct P
        # -- NOTE: For information about P, see the end this method's code
        Pdiag      = np.append(-1/self.DemB,np.zeros(Nvars-Nh)) # form diagonal of P
        self.theP  = -sp.diags(Pdiag,format="csc")              # form P: minus to maximize

        #  Construct q 
        # --- construct first Nh part of q for linear utility terms
        q1 = self.DemA/self.DemB
        # --- construct next Ngen part of q for marginal costs
        q2 = np.copy(-self.MarCost[0:Ngen])
        for _ in range(1,Nh):
            q2 = np.append(q2,-self.MarCost[0:Ngen])
        # end for time period loop

        # Construct q for rental of gen techs 
        q3 = np.copy(-self.CapCost[:-self.Nstor]) 

        # Construct q for rental of storage techs
        # --- NOTE: IF WE EVER ADD MARGINAL COSTS OF STORAGE, NEED TO CHANGE
        q4 = np.zeros(Nstor*(1+2*Nh))
        for i in range(0,self.Nstor):
            q4[i+i*2*Nh] = -self.CapCost[Ngen+i]
        # end for storage technology loop

        # Construct q by appending q1, q2, q3, and q4
        self.theq = -np.append(np.append(np.append(q1,q2),q3),q4).flatten()

        # P matrix- quadratic term for utility on main diagonal has -(1/Demb)
        # - utility is Q (DemA - 0.5Q)/DemB. note 0.5 is already accounted for in standard form
        # - P must be positive semi definite
        # - utility is Q*DemA/DemB - 0.5Q^2/DemB
        # -- 0.5 is already accounted for in standard form    
    # end create_Pq method
# end ModelCase subclass of ElectricityModel class

# class for different storage technology configurations
class ConfigureStorage:

    # initialize ConfigureStorage instance
    def __init__(self, ALPHA:ArrayOrNum, DELTA:ArrayOrNum, mod_case: ModelCase, NstorMax:int = 2) -> None:

        if ALPHA is None or DELTA is None:
            raise Exception('alphas and deltas must be supplied')
        else:
            ALPHA = np.atleast_1d(ALPHA).astype(float)
            DELTA = np.atleast_1d(DELTA).astype(float)
        # end if either ALPHA or DELTA is None

        if len(ALPHA) == len(DELTA):
            self.Nstor = len(ALPHA)
            if self.Nstor not in np.arange(1,NstorMax+1):
                errmss = 'Currently, can only accept at most ' + str(NstorMax) + ' storage technologies.'
                errmss = errmss + ' ' + str(self.Nstor) + ' technologies were supplied!'
                raise Exception(errmss)
            elif not self.Nstor == mod_case.Nstor:
                raise Exception('Number of storage parameters does not match the model case!')
            else:
                self.NstorMax = NstorMax
            # end if Nstor is less than NstorMax
        else:
            raise Exception('alphas and betas must have the same number of elements!')
        # end if/elif/else for infering Nstor
        
        if self.Nstor == 1:
            self.alphaS = ALPHA; self.deltaS = DELTA
            self.alphaL = 1;     self.deltaL = 0
        else:
            self.alphaS = ALPHA[0]; self.deltaS = DELTA[0]
            self.alphaL = ALPHA[1]; self.deltaL = DELTA[1]
        # end if/else for setting params 

        self.ALPHA = ALPHA
        self.DELTA = DELTA

        self.create_Aprime(mod_case)
    # end __init__ method

    # Create Case Invariant Matrices
    def create_Aprime(self, mod_case: ModelCase) -> None:

        # Store objects for cleaner code
        # --- Counts
        Nh     = mod_case.numhours  # Number of hours (numhours)
        Nvars  = mod_case.numvars   # Number of variables in model
        Ngen   = mod_case.Ngen      # Number of gen techs w/ MCs (Ngen)
        I_Nh   = np.identity(Nh)    # Nh by Nh identity matrix

        # --- Storage Techs
        Nstor    = self.Nstor    # Number of batteries/storage technologies
        NstorMax = self.NstorMax # Maximum number of storage technologies implimented

        # --- Vectors/Matrices
        Sp0Nh   = _Sparse0(Nh,Nh)
        zerovNh = np.zeros((Nh,1))
        Sp0vNh  = _Sparse0(Nh,1) 
        SpI_Nh  = _SparseI(Nh); 
        Sp1vNh  = sp.csc_matrix(np.ones((Nh,1))) 

        alphaS = self.alphaS; alphaL = self.alphaL
        deltaS = self.deltaS; deltaL = self.deltaL

        # If only one storage, increase Nvars as if Nstor was 2
        #  - See the end of this code for an explanation
        if Nstor == 1:
            Nvars += 1+2*Nh
        # end if 

        ############## CONSTRAINT 1
        # \sum_i qit  -Qt - \sum_m [1/(1-delta^m)]*S^m_t + [alpha^m/(1-delta^m)]*S^m_{t-1} + [1-(1+delta^m)/(1-delta^m)]*b^(m,d)_t \geq 0
        # i \in {1,...,Ngen}, m \in {s,l}

        # Initialize first Nvars-Nstor*(2*Nh+1) + 1 columns of A1 to be 0s (up to s^S_t)
        # --- Final A1 will be Nh by Nvars
        Atemp = np.zeros((Nh, Nvars - NstorMax*(2*Nh+1)+1))
        # --- Qt and qit part of the constraint
        for t in range(0,Nh):
            Atemp[t,t] = -1
            for i in range(0,Ngen):
                Atemp[t,Nh+Ngen*t+i] = 1
            # end for generation technology loop
        # end for each time period loop

        Atemp1 = sp.csc_matrix(Atemp)
        # SHORT DURATION PART
        # --- S^s_t and S^s_{t-1} part of
        Atemp = -1*I_Nh/(1 - deltaS)
        for t in range(1,Nh):
            Atemp[t,t-1] = alphaS/(1 - deltaS)
        # end for each time period loop
    
        # --- Add S_t and S_{t-1} part: hstack A1 and Atemp
        #A1 = np.hstack([A1,Atemp])
        # --- b^{s,d}_t part of the constraint
        Atemp2 = sp.csc_matrix(np.hstack([Atemp,(1 - ((1+deltaS)/(1-deltaS)))*I_Nh]))# remake Atemp
        # --- Add b^{s,d}_t part: hstack A1 and Atemp
        #A1 = np.hstack([A1,Atemp])
        # LONG DURATION PART 
        # --- \bar{S}^l part
        # See next line; it is the zerovNh column
        #Atemp3 = np.hstack([zerovNh])
        # --- S^l_t and S^l_{t-1} part
        Atemp = np.hstack([zerovNh,-1*I_Nh/(1 - deltaL)])
        for t in range(1,Nh):
            Atemp[t,t-1] = alphaL/(1 - deltaL)
        # end for each time period loop
        # --- Add S^l_t and S^l_{t-1} part
        #A1 = np.hstack([A1,Atemp])
        # --- b^{l,d}_t part
        Atemp3 = sp.csc_matrix(np.hstack([Atemp,(1 - ((1+deltaL)/(1-deltaL)))*I_Nh]))# remake Atemp
        # --- Add b^{l,d}_t part
        #A1 = np.hstack([A1,Atemp])
        # Make new A1 sparse
        A1 = sp.hstack([Atemp1,Atemp2,Atemp3],format='csc')
        del Atemp1, Atemp2, Atemp3, Atemp

        ############## CONSTRAINT 2
        # NOTE: Constraint 2 varies by case and is made in another function
        # For A2, need to hstack Nh*Ngen by Nh matrix

        ############## CONSTRAINT 3
        # s^S_t \geq 0, \all t \in 0:Nh
        A3S = sp.hstack([Sp0Nh, _Sparse0(Nh,Ngen*Nh+Ngen),Sp0vNh,SpI_Nh,Sp0Nh,Sp0vNh,Sp0Nh,Sp0Nh])

        ############## CONSTRAINT 4
        # \bar{S}^S - s^S_t \geq 0, \all t \in 0:Nh
        A4S = sp.hstack([Sp0Nh,_Sparse0(Nh,Nh*Ngen+Ngen),Sp1vNh,-1*SpI_Nh,Sp0Nh,Sp0vNh,Sp0Nh,Sp0Nh])

        ############## CONSTRAINT 5
        # Qt \geq 0, \all t \in 0:Nh
        A5 = sp.hstack([SpI_Nh,_Sparse0(Nh,Nh*Ngen+Ngen),Sp0vNh,Sp0Nh,Sp0Nh,Sp0vNh,Sp0Nh,Sp0Nh])

        ############## CONSTRAINT 6
        # q_it \geq 0, \all (i,t) \in 0:Ngen \times 0:Nh
        A6 = sp.hstack([_Sparse0(Nh*Ngen,Nh),_SparseI(Nh*Ngen),_Sparse0(Nh*Ngen,Ngen + NstorMax*(1+2*Nh))])

        ############## CONSTRAINT 7
        # K_i \geq 0, for all i \in 0:Ngen
        A7 = sp.hstack([_Sparse0(Ngen,Nh+Nh*Ngen),_SparseI(Ngen),_Sparse0(Ngen,NstorMax*(1+2*Nh))])

        ############## CONSTRAINT 8
        # \bar{S}^S \geq 0
        A8S = sp.hstack([_Sparse0(1,Nh+Nh*Ngen+Ngen),_SparseI(1),_Sparse0(1,NstorMax*(1+2*Nh)-1)])

        ############## CONSTRAINT 9
        # b^{S,d}_t \geq 0, \all t \in 0:Nh
        A9S = sp.hstack([_Sparse0(Nh, Nh+Nh*Ngen+Ngen+1+Nh),SpI_Nh,_Sparse0(Nh,1+2*Nh)])

        ############## CONSTRAINT 10
        # b^{S,c}_t \geq 0, for all t\in 0:Nh
        # Reminder: b^c_t = 1/(1-delta)*S_t - alpha/(1-delta)*S_{t-1} + (1+delta)/(1-delta)*b^d_t    

        #  SHORT DURATION PART
        # --- Zeros up until s^S_t
        Atemp1 = _Sparse0(Nh, Nvars-NstorMax*(2*Nh+1)+1)
        # --- Form s^S_t and s^S_{t-1} part
        Atemp = (1/(1-deltaS))*I_Nh
        for t in range(1,Nh):
            Atemp[t,t-1] = -alphaS/(1 - deltaS)
        # end for each time period loop
        # --- Form b^{S,d}_t part of A10 
        Atemp2 = sp.csc_matrix(np.hstack([Atemp,((1+deltaS)/(1-deltaS))*I_Nh]))
        # --- hstack together and end with zeros for long duration storage variables
        A10S = sp.hstack([Atemp1,Atemp2,_Sparse0(Nh,1+2*Nh)],format='csc')
        del Atemp, Atemp1, Atemp2

        # If Nstor is 1, we are finished and can combine constraint matrices
        if self.Nstor == 1:
            Aprime = sp.vstack([A1,A5,A6,A7,A3S,A4S,A8S,A9S,A10S],format="csc")
            Aprime = Aprime[:,:mod_case.numvars]
        # Otherwise (Nstor == 2), 
        # Make Versions of Constraints 3, 4, 8, 9, and 10 for Long Duration Storage
        # and then combine all matrices
        else:
            ######## Constraint 3
            # s^L_t \geq 0, \all t \in 0:Nh
            A3L = sp.hstack([Sp0Nh, _Sparse0(Nh,Ngen*Nh+Ngen),Sp0vNh,Sp0Nh,Sp0Nh,Sp0vNh,SpI_Nh,Sp0Nh])
            ######## Constraint 4
            # \bar{S}^L - s^L_t \geq 0, \all t \in 0:Nh
            A4L = sp.hstack([Sp0Nh,_Sparse0(Nh,Nh*Ngen+Ngen),Sp0vNh,Sp0Nh,Sp0Nh,Sp1vNh,-1*SpI_Nh,Sp0Nh])
            ######## Constraint 8
            # \bar{S}^L \geq 0
            A8L = sp.hstack([_Sparse0(1,Nh+Nh*Ngen+Ngen),_Sparse0(1,1+2*Nh),_SparseI(1),_Sparse0(1,2*Nh)])
            ######## Constraint 9
            # --- b^{L,d}_t \geq 0, \all t \in 0:Nh
            A9L = sp.hstack([_Sparse0(Nh, Nvars-Nh),SpI_Nh])

            ######## Constraint 10
            # b^{S,c}_t \geq 0, for all t\in 0:Nh
            # Reminder: b^c_t = 1/(1-delta)*S_t - alpha/(1-delta)*S_{t-1} + (1+delta)/(1-delta)*b^d_t    
            # --- Zeros up until s^L_t
            Atemp1 = _Sparse0(Nh,Nvars-2*Nh)
            # --- Form s^L_t and s^L_{t-1} part 
            Atemp = (1/(1-deltaL))*I_Nh
            for t in range(1,Nh):
                Atemp[t,t-1] = -alphaL/(1 - deltaL)
            # --- Form b^{L,d}_t part 
            Atemp2 = sp.csc_matrix(np.hstack([Atemp,((1+deltaL)/(1-deltaL))*I_Nh]))
            # --- Form A10L by hstacking Atemp1 and Atemp2
            A10L = sp.hstack([Atemp1,Atemp2],format='csc')
            del Atemp, Atemp1, Atemp2

            Aprime = sp.vstack([A1,A5,A6,A7,A3S,A4S,A8S,A9S,A10S,A3L,A4L,A8L,A9L,A10L],format="csc")
        # end if number of storage technologies 

        self.Aprime = Aprime

        # Explanation of code:
        # I have set up this model so that the storage technologies are at the 
        # end of the variable list and each storage technology has the following
        # variables in this order: Kstor (1), St (numhours), bdt (numhours).
        # As such, if we assume that the short duration (battery) is used when 
        # there is only one storage technology (which the __init__ method for
        # ConfigureStorage explicitly assumes is the case), we can reuse this 
        # code for both Nstor == 1 and Nstor == 2 and simply ignore the long 
        # duration variables and constraints. Note that in the last if statement,
        # this code, Aprime[:,:mod_case.numvars], is removing the long duration
        # variables.

        # Derivation of Constraint 1:
        # LOM: S_t = \alpha S_{t-1} + (1-\delta)b^c_t - (1+\delta)b^d_t
        # Rewrite LOM: b^c_t = (1/(1-delta))S_t - (alpha/(1-delta))S_{t-1} + ((1+delta)/(1-delta))b^d_t
        # Resource Constraint: Qt + b^c_t - b^d_t \leq \sum_i q_it
        # Sub in for b^c_t into RC: Qt + (1/(1-delta))S_t - (\alpha/(1-delta))S_{t-1} + ((1+delta)/(1-\delta))b^d_t - b^d_t \leq \sum_i q_it
        # or Qt + (1/(1-delta))S_t - (alpha/(1-delta))S_{t-1} -(1 - (1+delta)/(1-delta))b^d_t \leq \sum_i q_it
        # or \sum_i q_it - Qt - (1/(1-delta))S_t + (alpha/(1-delta))S_{t-1} + (1-(1+delta)/(1-delta))b^d_t \geq 0
    # end create_Aprime method
# end ConfigureStorage class

class ModelSolver:

    # initialize ModelSolver instance
    def __init__(self, tol_inf:float = 1e-14, tol_grad:float = 0.1, tol:float = 1e-10,
                 trim_tol:float = 1e-8,max_iter:int = 1e6, ub_val:float = 1e11,
                 tol_abs:Optional[float] = None, tol_rel:Optional[float] = None, 
                 equ_cons:Optional[ArrayLike]=None, **options) -> None:
        
        # if either tol_rel or tol_abs is None, assign them values
        if tol_rel is None or tol_abs is None:
            # if tol is not None, assign it to any None object
            if tol is not None:
                if tol_abs is None:
                    tol_abs = tol
                # end if 
                if tol_rel is None:
                    tol_rel = tol
                # end if 
            # else, tol is None
            else:
                # if tol_abs and tol_rel are also none, throw an error
                if tol_abs is None and tol_rel is None:
                    raise Exception('At least one of tol, tol_abs, and tol_rel must be supplied')
                # else, assign the value of the non-None object to the None object
                else:
                    # ADD WARNING HERE LATER
                    if tol_abs is not None and tol_rel is None:
                        tol_rel = tol_abs
                    else:
                        tol_abs = tol_rel
                    # end if/else 
                # end if/else
            # end if/else

        # If WarmStart 
        if 'warm_start' in options and options['warm_start']:
            if 'x0' in options and 'y0' not in options:
                options['y0'] = None
            elif 'y0' in options and 'x0' not in options:
                options['x0'] = None
            else:
                raise TypeError('User indicated WarmStart but did not supply x0 or y0')
            # end if/elif/else
        # end if warm start enabled

        # store settings for model settings in options dictionary and 
        # add it to instances
        options['ub_val']   = ub_val
        options['tol_abs']  = tol_abs
        options['tol_rel']  = tol_rel
        options['tol_inf']  = tol_inf
        options['tol_grad'] = tol_grad
        options['trim_tol'] = trim_tol
        options['max_iter'] = int(max_iter)
        options['equ_cons'] = equ_cons
        self.options        = options
    # end __init__ method

    # Update initial x and y for warm start
    def update_warm_start(self,x:Optional[ArrayLike] = None,y:Optional[ArrayLike] = None) -> None:

        # enable warm start
        self.options['warm_start'] = True 
        
        # if at least one of x or y is supplied, update the supplied vectors 
        if x is not None or y is not None:
            
            if x is not  None:
                self.options['x0'] = x
            # end if x is None

            if y is not None:
                self.options['y0'] = y
            # end if y is none

        # else, disable warm start
        else: 
            self.options['warm_start'] = False
        # end if/else either x or y is not None
    # end upade_warm_start method

    # update osqp solver options to match model options
    def update_solver_options(self) -> None:

        # assign model settings to their corresponding solver settings
        self.solver_options['eps_abs']      = self.options['tol_abs']
        self.solver_options['eps_rel']      = self.options['tol_rel']
        self.solver_options['eps_prim_inf'] = self.options['tol_inf']
        self.solver_options['eps_dual_inf'] = self.options['tol_inf']

        # if max_iter is in (model) options but not in solver_options, 
        #   assign max_iter from options to max_iter in solver_options
        if not 'max_iter' in self.solver_options:
            self.solver_options['max_iter'] = self.options['max_iter']
        # end if max_iter not in solver_options
    # end update_solver_options method

    def guess_solution(self,mod_case: ModelCase):
            
            DF     = mod_case.DF.iloc[mod_case.hIDs]
            Nh     = mod_case.numhours
            Ngen   = mod_case.Ngen
            Nrenew = mod_case.Nrenew
            Ncon   = self.theA.shape[0]

            # Form initial guess of x (optimal choices)
            Qt = np.array(DF['demand']-DF['ngwat'])+mod_case.demandinc
            
            fs  = np.ones((Nh,Ngen - Nrenew))
            fs  = np.hstack([np.array(DF[['cfsun','cfwind']]),fs])
            qws = fs/np.repeat(np.sum(fs,1),Ngen).reshape((Nh,Ngen))

            qit = np.tile(Qt.reshape((Nh,1)),(1,Ngen)) * qws
           
           # Ks = np.tile(np.max(np.max(qit,0)),(Nh,1))
            #x0 = np.hstack([Qt,qit.flatten(),Ks[0,:]])
            s0 = np.max(Qt)*0.05
            Ks = np.tile(np.max(qit,0).flatten(),(Nh,1))
            x0 = np.hstack([Qt,qit.flatten(),Ks[0,:],s0])
            x0 = np.append(x0,np.zeros(self.theA.shape[1]-x0.shape[0]))
            # Qt (numhours); qit (Ngen)*numhours); Ki (Ngen);      Sbar (1); St (numhours); b^d_t (numhours)
            x0[Nh+Nh*Ngen+Ngen+1+2*Nh] = s0
            
            # Form initial guess of y (lagrangian multipliers)
            ps = np.tile(np.array(DF['price']).reshape((Nh,1)),(1,Ngen))
            cs = np.tile(mod_case.MarCost[:Ngen],(Nh,1))

            y0 = np.zeros(qit.shape)
            y0[qit==Ks] = (ps-cs)[qit==Ks]
            y0[qit < self.options['tol_inf']] = float(0)
            y0 = y0.flatten()
            y0 = -np.hstack([y0,ps[:,0],np.zeros(Ncon-Nh*Ngen-Nh)])

            return x0, y0

    
    # setup model solver interface
    def setup(self, mod_case: ModelCase, stor_config:ConfigureStorage, **solver_options) -> osqp.interface:

        # store solver options and update solver options to match default options
        self.solver_options = solver_options
        self.update_solver_options()

        # create the A matrix by vertically stacking A2 and Aprime
        self.theA = sp.vstack([mod_case.A2,stor_config.Aprime],format='csc')
        
        # create lower and upperbounds
        self.lb = np.zeros(self.theA.shape[0])
        self.ub = np.ones(self.theA.shape[0])*self.options['ub_val']

        if self.options['equ_cons'] is not None:
            self.ub[self.options['equ_cons']] = float(0)

        # initialize OSQP interface and setup solver
        OSQP_prob = osqp.OSQP()
        OSQP_prob.setup(mod_case.theP,mod_case.theq,self.theA,self.lb,self.ub,**self.solver_options)

        # if warm_start is in options and if its true, set initial x and y
        if 'warm_start' in self.options and self.options['warm_start']:
            OSQP_prob.warm_start(x=self.options['x0'],y=self.options['y0'])
        #else:
            #x0, y0 = self.guess_solution(mod_case)
            #OSQP_prob.warm_start(x=x0,y=y0)   
        # end if warm_start is True

        # return setup OSQP solver
        return OSQP_prob
    # end setup method

    def solve(self, OSQPmodel: OSQP_interface):
        return OSQPmodel.solve()
    # end solve method
# end ModelSolver class


# Store Solution To Battery Problem
# --- sol has ### choice vector Q_t (numhours) ; q_it (NT)*numhours) ; Ki (NT); Sbar (1); S_t (numhours); b^d_t (numhours)
class BatterySolution(ElectricityModel):

    # initialize BatterySolution instance
    def __init__(self,mod_case:ModelCase, stor_config:ConfigureStorage, solver:ModelSolver, OSQPsol:osqp.interface):

        # Store Needed Objects from ModelCase
        self.case      = mod_case.case       # DataFrame of Case Parameters
        self.hIDs      = mod_case.hIDs       # IDs for shifting data to battery start hour
        self.DemA      = mod_case.DemA       # Recovered demand constants
        self.DemB      = mod_case.DemB       # Recovered demand slopes
        self.genVars   = mod_case.genVars    # generation techs
        self.renewVars = mod_case.renewVars  # renewable generation techs
        self.MarCost   = mod_case.MarCost    # marginal costs
        self.CapCost   = mod_case.CapCost    # capital costs
        self.Ngen      = len(self.genVars)   # number of generation techs
        self.Nrenew    = len(self.renewVars) # number of renewable generation techs

        # Store Storage Parameters
        self.ALPHA = stor_config.ALPHA       # self discharge parameters
        self.DELTA = stor_config.DELTA       # transmission loss parameters
        self.Nstor = len(stor_config.ALPHA)  # number of storage technologies
        
        # Store OSQP solver variables
        self.iter = OSQPsol.info.iter # number of iterations
        self.xopt = OSQPsol.x         # optimal x (primal choice variables)
        self.yopt = OSQPsol.y         # optimal y (dual choice variables)

        # Store Solver Settings
        self.tol_grad = solver.options['tol_grad'] # tolerance for gradient
        self.max_iter = solver.options['max_iter'] # max number of iterations
        self.tol_abs  = solver.options['tol_abs']  # absolute tolerance
        self.tol_rel  = solver.options['tol_rel']  # relative tolerance
        self.tol_inf  = solver.options['tol_inf']  # tolerance for infeasibility
        self.trim_tol = solver.options['trim_tol'] # tolerance for infeasibility
        self.ub_val   = solver.options['ub_val']   # upperbound of variables
        
        # Format solution and form model objects
        self.format_solution()
        self.format_capacities()
        self.calc_electricity_gen()
        self.calculate_outcomes()
        self.convergence_check()

    # end __init__ method

    # Format solution to OSQP into pandas DataFrames
    def format_solution(self):

        trim_tol = self.trim_tol

        # set numhours, Ngen, and Nstor for cleaner code
        # --- model "counts"
        Nh = self.numhours; Ngen = self.Ngen; Nstor = self.Nstor
        # --- storage parameters
        alphas = self.ALPHA; deltas = self.DELTA
        # --- model solution vectors
        xopt = self.xopt; yopt = self.yopt

        # trim values of xopt and yopt "close to zero"
        xopt[xopt <= trim_tol]         = float(0)
        yopt[np.abs(yopt) <= trim_tol] = float(0) # note: yopt <= 0
        
        # Create vectors of names for variables 
        # --- set names of the variables needed to export without prefix/suffix
        storvars = ['Kstor','st','bdt','bct']
        # --- add "K" (capacity/capital) prefix to capvars
        capvars = ['K'+v for v in self.genVars]
        # --- add "S" and "L" (short and long) suffix to storvars
        storvarsDur = [v + s for v in storvars for s in ['S','L']]
        # --- add capacity storage vars to capvars vector
        capvars = capvars + [v for v in storvarsDur if v.startswith("Kstor")]
        # --- pull off capacity storage vars from stordurvars
        storvarsDur = [v for v in storvarsDur if not v.startswith("Kstor")]
        # --- if Nstor is one, remove L (long) duration storage
        if Nstor == 1:
            capvars     = [v for v in capvars if not v.endswith("L")]
            storvarsDur = [v for v in storvarsDur if not v.endswith("L")]
        # end if one storage technology

        # set non-storage variables  
        pt   = np.abs(yopt[Ngen*Nh:Ngen*Nh+Nh])
        yid0 = Ngen*Nh+Nh+Nh+Nh*Ngen+Ngen+Nh
        Qt  = xopt[0:Nh]  # quantity demanded
        qit = xopt[Nh:Nh+Nh*Ngen]             # quantity generated from techs
        ks  = xopt[Nh+Ngen*Nh:Nh+Ngen*Nh+Ngen] # capacities for generation techs
        xid0 = Nh + Ngen * Nh + Ngen           # set last value of xopt indexed

        # initialize arrays to store storage duration variables
        sbar = np.empty(Nstor);      st  = np.empty((Nh,Nstor))
        bdt  = np.empty((Nh,Nstor)); bct = np.empty((Nh,Nstor))
        mut  = np.empty((Nh,Nstor))

        # loop over each storage duration
        for k in range(0,Nstor):
            # pull off corresponding langranian multipliers
            mut[:,k] = np.abs(yopt[yid0:yid0+Nh])
            yid0 += Nh + 1 + 3*Nh

            # pull off corresponding values from OSQP solution vector
            sbar[k]  = xopt[xid0]                   # storage capacity
            st[:,k]  = xopt[xid0+1:xid0+1+Nh]       # storage state 
            bdt[:,k] = xopt[xid0+1+Nh:xid0+1+Nh+Nh] # storage draw
            xid0    += 1+Nh+Nh                      # update last value of xopt indexed
            # Recover b^c_t from S_t and b^d_t
            # - Note:  b^c_t = S_t/(1-delta) - alpha/(1-delta)*S_{t-1} + (1+delta)/(1-delta)*b^d_t
            bct[:,k]  = (st[:,k]+(1+deltas[k])*bdt[:,k])/(1-deltas[k])
            bct[1:,k] = bct[1:,k]-alphas[k]*st[:-1,k]/(1-deltas[k])
        # end for k loop (over storage technologies)

        # create DataFrame of capacities (NOTE: Will have one observations)
        caps = np.append(ks,sbar)
        cDF  = pd.DataFrame(caps.reshape((1,Ngen+Nstor)),columns=capvars)

        # create DataFrame of all time varying variables (NOTE: Will have numhours observations)
        casev   = self.case['caseID']*np.ones(Nh) # vector of the case number repeated
        hourv   = np.arange(1,Nh+1)                # vector of each hour: 1 to Nh

        tDFvars = ['caseID'] + ['hour'] + self.genVars + ['Qt'] + storvarsDur + ['pt'] + ['mutS']

        if self.Nstor == 2:
            tDFvars = tDFvars + ['mutL']

        qit = qit.reshape((Nh,Ngen))
        tDF = np.column_stack([casev,hourv,qit,Qt,st,bdt,bct,pt,mut])
        tDF = pd.DataFrame(tDF,columns=tDFvars)

        self.timeDF   = tDF
        self.capDF    = cDF
        self.minprice = min(min(pt), self.ub_val)
        self.maxprice = max(max(pt), trim_tol)
        self.batutil  = np.sum(np.abs(bct-bdt),0)/2  # CHECK THIS
    # end format_solution

    #  Create Capacity Variables (Ks)  
    def format_capacities(self) -> None:

        # store objects for cleaner code
        Nh = self.numhours; Ngen = self.Ngen; Nrenew = self.Nrenew
        cfmat = np.array(self.DF[['cfsun','cfwind']])[self.hIDs,:]
        unitvec = np.ones(Nh)

        # Create Kvec and trim small K's
        Kvec = np.abs(np.array(self.capDF.iloc[0])) # create Kvec
        
        # Initialize KStepMat and Set First Step Using Kvec for Renewables
        KStep1        = np.matmul(Kvec[:Nrenew],cfmat.T)
        KStepMat      = np.zeros((Nh,Ngen-Nrenew+1))
        KStepMat[:,0] = np.copy(KStep1)
        
        # Add cols to Kstep mat for each non-renewable tech
        for i in range(0, Ngen - Nrenew):
            KStepMat[:,i+1] = KStepMat[:,i] + Kvec[Nrenew+i]*unitvec
        # end for i loop (over non-renewable techs)
        
        # Store K Variables in Solution Object
        self.Kvec = Kvec; self.KStep = KStepMat; self.KStep1 = KStep1
    # end format_capacities method

    # Calculate how much electricity each technology generates
    def calc_electricity_gen(self) -> None:

        # store objets for cleaner code
        DF = self.DF.iloc[self.hIDs]
        Nh = self.numhours; 
        tDF = self.timeDF; cDF = self.capDF
        bdvars = [s for s in tDF.columns if s.startswith("bdt")]
        bcvars = [s for s in tDF.columns if s.startswith("bct")]

        # Calculate and Store Hydro Measures
        ngwat = DF.ngwat
        self.genHydro = sum(ngwat)
        self.maxHydro = np.amax(ngwat)

        # Calculate Non-Renewable Measures
        # --- Initialize generation vectors for gas techs, renewables, and nuke
        genGasCCbig = np.zeros(Nh) # gas, combined cycle 
        genGasPbig  = np.zeros(Nh) # gas, peak
        genNukebig  = np.zeros(Nh) # nuke      

        # --- Store qit's into intialized vectors
        for t in range(0,Nh):
            genGasPbig[t]  = tDF.loc[t].loc['gasP']
            genGasCCbig[t] = tDF.loc[t].loc['gasCC']
            genNukebig[t]  = tDF.loc[t].loc['nuke']
        # end for ecah time period loop 

        # --- zero out if small K
        genNuke   = sum(genNukebig)  * (cDF.loc[0].loc['Knuke'] != 0)
        genGasCC  = sum(genGasCCbig) * (cDF.loc[0].loc['KgasCC'] != 0)
        genGasP   = sum(genGasPbig)  * (cDF.loc[0].loc['KgasP'] != 0)

        self.genNonRen = np.array([genNuke,genGasCC,genGasP])
        # Store Generation Measures in Solution Obkect
        self.genNuke = genNuke; self.genGasCC = genGasCC; self.genGasP = genGasP
        
        # Renewables Generation  
        # --- Generate temporary sun and wind generations
        genSuntemp  = cDF.loc[0].loc['Ksun']*np.array(DF.cfsun)
        genWindtemp = cDF.loc[0].loc['Kwind']*np.array(DF.cfwind)
        # --- Calculate Curtailment for Sun and Wind
        curtailsun = np.zeros(Nh); curtailwind = np.zeros(Nh) # init. vectors
        bdt = np.array(tDF[bdvars]); bct = np.array(tDF[bcvars])
        
        if (sum(genWindtemp) + sum(genSuntemp)) > 0:
            
            # only need this for linear demands
            genDiff  = np.clip(np.array(self.DemA) + np.sum(bct-bdt,1) - self.KStep1,None,0) # CHECK THIS
            
            # Calculate Fraction of Renewable Generation from Sun and Wind
            fracwind = np.zeros(Nh); fracsun = np.zeros(Nh) # init. sun & wind shares
            
            for t in range(0,Nh):
                if genWindtemp[t] + genSuntemp[t] > 0:
                    fracwind[t] = (genWindtemp[t])/((genWindtemp[t]+genSuntemp[t]))
                    fracsun[t]  = 1-fracwind[t]
                else:
                    fracwind[t] = 0; fracsun[t] = 0
                # end if/else 
            # end for time period loop
            
            # Calculate Sun and Wind Curtailment 
            curtailwind = fracwind*genDiff; curtailsun = fracsun*genDiff
        
        # Calcaulate Sun and Wind generation and total curtailment
        # --- curtailsun and curtailwind are negative
        self.genSun  = sum(genSuntemp)  + sum(curtailsun)
        self.genWind = sum(genWindtemp) + sum(curtailwind)
        self.curtail = sum(curtailsun)  + sum(curtailwind)
    # end calc_electricity_gen method

    # Calculate market/observable outcomes (welfare, profits, emissions, etc)
    def calculate_outcomes(self) -> None:

        # store objects for cleaner code
        tDF  = self.timeDF; DF = self.DF.iloc[self.hIDs]; DemB = self.DemB
        cfmat = np.array(DF[['cf'+s for s in self.renewVars]])
        capDF = self.capDF; Kvec = np.array(capDF.iloc[0]); pt = np.array(self.timeDF.pt)
        # --- set technology names and storage names
        techVars    = self.capDF.columns
        storVars    = [v for v in techVars if v.startswith('Kstor')]
        nonrenewIDs = [s[1:] for s in techVars]
        nonrenewIDs = [v in self.renewVars or v.startswith('stor') for v in nonrenewIDs]
        nonrenewIDs = np.where([not v for v in nonrenewIDs])

        # Add hydro back into demand to properly calculate total utility
        consumpt = np.array(tDF.Qt) + np.array(DF.ngwat)   # into consumption
        DemA     = np.array(self.DemA) + np.array(DF.ngwat) # into demand curve constants
        
        # Calculate total nonrenewable generation costs and total capital costs
        costNonRen = sum(np.multiply(self.genNonRen,self.MarCost[nonrenewIDs]))
        capcost    = sum(np.multiply(Kvec,self.CapCost)) 
        
        # Calculate revenue, total consumption, utility, and total surplus
        revenue = sum(np.multiply(pt,consumpt))       # electricity revenues
        consume = sum(consumpt)                       # total consumption
        utility = consumpt*(DemA-0.5*consumpt)/DemB   # consumer utility
        surplus = np.sum(utility)-costNonRen-capcost  # total surplus
        
        # Calculate emissions: conversion is in metric tons/MWh -> convert to metric tons/GWH
        emissions  = self.genGasP*self.CO2GasP*self.GWconv   # Gas: Peak
        emissions += self.genGasCC*self.CO2GasCC*self.GWconv # Gas: Combined Cycle
        
        # Initialize vector to store profits of each technology
        newprof = np.zeros(Kvec.shape)

        # Calculate profits for each technology
        for i, v_i in enumerate(techVars):

            # if v_i is a storage technology variable, calc profits of storage tech
            # else, calculate profits from generation
            if v_i in storVars:
                dur = v_i[-1] # storage "duration" (S or L for short or long)

                # if Kvec is not 0, calculate profits
                # else, use max/min price difference
                if not isclose(Kvec[i],float(0)):
                    newprof[i] = np.sum(-self.timeDF['pt']*(self.timeDF['bct'+dur]-self.timeDF['bdt'+dur]))/Kvec[i] # THIS ONE
                else:
                    newprof[i] = self.maxprice - self.minprice
                # end if storage technology is used
            else:
                if v_i[1:] in self.renewVars:
                    newprof[i] = sum(np.maximum(pt-self.MarCost[i],0)*cfmat[:,i])
                else:
                    newprof[i] = sum(np.maximum(pt-self.MarCost[i],0))
                
                
                # end if tech is renewable
            # end if a storage technology
        # end foor loop over technologies

        # Store Calculated Market Outcomes in Solution Object
        self.costNonRen = costNonRen; self.capcost = capcost # costs
        self.revenue = revenue; self.consume = consume;      # rev and consume
        self.surplus = surplus; self.emissions = emissions   # surplus & emissions
        self.newprof = newprof                               # newprof
    # end calculate_outcomes method

    # Check if model solution converged
    def convergence_check(self) -> None:

        # store capacity DataFrame as np array 
        Kvec = np.array(self.capDF.iloc[0])

        # see if solution form LQ solver converges in the gradient
        gradvec = (self.newprof - self.CapCost) # Create gradient vector
        gradvec *= 1 / float(self.GWconv)         # convert gradvec back to $/MW

        # Test for zero profit condition
        # The zero profit criterion is satisfied when the FOCs hold
        # which means either 
        #  1) K_i is in the interior (K_i>0) and its derivative is 0
        #  2) K_i is on the boundary (K_i==0) and its derivative <= 0
        isK0      = np.array([isclose(K_i,0) for K_i in Kvec])
        isPosK    = np.logical_not(isK0)
        gradSmall = np.abs(gradvec) <=  self.tol_grad
        gradNeg   = gradvec <= float(0)
        
        # Test for convergence for interior and boundary cases
        interiorTest = isPosK & gradSmall
        boundaryTest = isK0 & gradNeg

        # Test each K_i for convergence (only one must be true)
        convergeTest = interiorTest | boundaryTest
        
        # If all techs meet convergence criteria, converged in zero profit condition
        if np.sum(convergeTest) == Kvec.size:
            convergence = 1
        else:
            # Some techs didn't converge but NIt < MaxIt, OSP converged
            if self.iter < self.max_iter:
                convergence = 2
            # Some techs didn't converge and NIt == MaxIt, OSQP didn't converge
            else:
                convergence = 3
            # end second convergence check
        # end convergence check

        # Store convergence ID and gradient
        self.convergence = convergence
        self.gradvec     = gradvec
        # CONVEGENCE CRITERIA: (1) is FOC, (2) is complementary slackness 
        #   1) Tech is used (K_i > 0): abs(grad_i) < tolerance \approx 0 
        #   2) Tech is not used (K_i == 0): grad_i <= 0 
    # end convergence_check method

    # Write Time Series Data (timeDF) to CSV
    def export_hourly_data(self, fname='temp_hourly_output.csv',out_path:Union[str,None]=None) -> None:

        if out_path is None:
            out_path=self.PATH
        else:
            if out_path[-1] != '/':
                out_path = out_path + '/'

        self.timeDF.to_csv(out_path+fname,index=False)
    # end export_hourly_data

    # Write Summary of Numerical Solution to CSV
    def export_solution_summary(self,fname='temp_output.csv',out_path:Union[str,None]=None) -> None:
       
        if out_path is None:
            out_path=self.PATH
        else:
            if out_path[-1] != '/':
                out_path = out_path + '/'


        SRME      = -9999
        GWconv    = self.GWconv
        pricestd  = np.std(self.timeDF.pt)/GWconv
        answervec = np.array(self.convergence)
        itervec   = np.array(self.iter)
        gradvecnp = np.array(self.gradvec)

        genVars = [self.genSun, self.genWind,self.genHydro,self.genNuke,self.genGasCC,self.genGasP]
        genVars = np.array(genVars)*GWconv
       
        outvec  = np.array(self.Kvec*GWconv)
        outvecL = np.append(np.array([self.revenue, self.consume]),genVars)
        outvecL = np.append(outvecL,
                            np.array([self.surplus, self.emissions/1e6,
                            self.minprice/GWconv,self.maxprice/GWconv,SRME, 
                            self.curtail*GWconv]))

        outvec = np.append(outvec,outvecL)
        outvec = np.append(outvec,answervec)
        outvec = np.append(outvec,gradvecnp)
        outvec = np.append(outvec,np.array(self.case))
        outvec = np.append(outvec,pricestd)
        outvec = np.append(outvec,self.maxHydro*GWconv)
        outvec = np.append(outvec,self.batutil*GWconv)
        outvec = np.append(outvec,itervec)

        outvars = ["KSol", "KWin", "KNuk", "KGasCC", "KGasP", "KBat", "KLDS", 
                   "revenue", "consumpt", "genSol", "genWin", "genHydro", "genNuk", 
                    "genGasCC", "genGasP", "surplus", "carbon", "minprice", "maxprice", 
                    "SRME", "curtail", "answer", "gradS", "gradW", "gradN", "gradCC", 
                    "gradP", "gradB", "gradL","caseID", "SunInnov", "WindInnov", "NukInnov", 
                    "carbontax", "BatInnov", "LongInnov", "batstarthour", "DatCentInc", 
                    "priceSD", "KHydro", "BatUtil", "LDSUtil", "iterations"]
        
        outvars = np.array(outvars)

        if self.Nstor == 1:

            isLongVar = [v.startswith(('gradL','LDS')) or v.endswith(('LDS','Long')) for v in outvars]
            isLongVar = np.array(isLongVar)
            outvars   = outvars[np.where(np.logical_not(isLongVar))]
            
       
        outDF = pd.DataFrame(outvec.reshape((1,len(outvec))),columns=outvars)

        outDF.to_csv(out_path+fname,index=False)

        # end with data export
    # end export_solution_summary method
# end BatterySolution subclass of ElectricityModel class

    
