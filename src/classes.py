import numpy as np
from numpy import nan as NAN
from numpy.typing import ArrayLike

import pandas as pd
from pandas._libs.missing import NAType, checknull, NA

from typing import Union, Optional

from osqp import interface as osqp_interface
from dataclasses import dataclass, field, fields, asdict
from math import isclose


type Vec = Union[ArrayLike,float,None]
type OSQP_interface = Union[osqp_interface.OSQP,None]

    
def _2nan(x:ArrayLike):

    # if x is a scalar, check if it is not a null value
    if isinstance(x,Union[float,int,NAType,None]):
        if checknull(x):
            x = NAN
    # if x is an array-like object, loop over each element & check if null value
    else:
        for i, x_i in enumerate(x):
            if checknull(x_i):
                x[i] = NAN
    return x


def _class2str(x):
    return str(x).replace('<class \'','').replace('\'>','')

def _2float(x,Types=(float,int)):

    x = _2nan(x) # convert None's and NA's to NaNs which are floats


    if not (isinstance(x,Types) or all(isinstance(xi, Types) for xi in x)):

        xStr    = [_class2str(x) for x in set([type(xi) for xi in x])]
        TypeStr = ', '.join([_class2str(ti) for ti in Types])

        raise ValueError('Elements of x must be the following types: ' + 
                         TypeStr+ '\n   Unique types in x: ' + xStr)
    # end 


    if isinstance(x,Types):
        x = float(x)
    elif type(x) is np.ndarray:
        x = x.astype(float)
    elif type(x) in [tuple,list]:
        x = np.array(x,dtype=float)
    else:
        raise ValueError('Cases doesn\'t catch x of type '+_class2str(type(x)))
    
    return x

# =========================================================================== #
#                      DATACLASSES FOR TECHNOLOGIES                           #
# =========================================================================== #

# --- Storage Technology
@dataclass(frozen=True,order=True)
class StorTech:
    '''dataclass to define storage technologies'''
    _sort_key: tuple = field(init=False, repr=False) # sort key
    capcost:  float                                  # fixed/capital cost
    margcost: float = 0.0                            # marginal cost
    alpha:    float = 1.0                            # self-discharge rate
    delta:    float = 0.0                            # transmission loss

    # post initialization
    def __post_init__(self):

        # coece all fields of type float to float
        for key, val in asdict(self).items():
            object.__setattr__(self, key, _2float(val))

        # set sortkey order for dataclass to sort storage techs
        object.__setattr__(self, '_sort_key', (self.margcost,self.capcost,
                                               self.alpha,self.delta))


# --- Electricity Generation Technology
@dataclass(frozen=True,order=True)
class GenTech:
    '''dataclass to define electricity generation technologies'''
    margcost: float                     # marginal cost of generation
    capcost:  float                     # fixed/capital cost of technology
    co2:      float                     # CO2 emissions from generation
    renew:    bool  = field(init=False) # is technology renewable?
    carbon:   bool  = field(init=False) # does technology emit carbon?

    def __post_init__(self):

        carbon:bool = self.co2 > 0.0
        renew:bool  = not carbon and isclose(self.margcost,0.0)

        object.__setattr__(self, 'carbon', carbon)
        object.__setattr__(self, 'renew', renew)


# =========================================================================== #
#                                                                             #
# =========================================================================== #

@dataclass(frozen=True)
class Case:
    '''dataclass for a specific case of the scenarios'''
    RenewInv:   float = 0.0             # % decrease in renewable gen tech
    CarbonTax:  float = 0.0             # carbon tax
    DatCentInc: float = 0.0             # data center hourly demand increment
    ShortInv:   float = 0.0             # cost innov of short dur storage 
    LongInv:    float = 0.0             # cost innov of long dur storage
    Nstor:      int = field(init=False) # number of storage technologies
    # NOTE: ADD 'collections' of baseline storage and generation technologies here

    def __post_init__(self):

        # coece all fields of type float to float
        for fld in fields(self):
            # if field type is float, make sure it is float
            if fld.type is float:
                key = fld.name
                val = getattr(self,key)
                object.__setattr__(self, key, _2float(val))
            # end if field type is float
        # end loop over fields

        # calculate number of storage technologies
        Nstor = (not checknull(self.ShortInv)) + (not checknull(self.LongInv))
        object.__setattr__(self, 'Nstor', Nstor) 



@dataclass(frozen=True)
class PossibleCases:
    '''dataclass for possible cases to consider'''
    RenewInv:   Vec = 0.0 # % decrease in renewable gen techs
    CarbonTax:  Vec = 0.0 # carbon tax \in [0,\infty]
    DatCentInc: Vec = 0.0 # hourly electricity demand boost from data centers
    ShortInv:   Vec = 0.0 # % decrease in short duration storage tech
    LongInv:    Vec = 0.0 # % decrease in long duration storage tech

    def __post_init__(self):
        # coece all fields of type float to float
        for key, val in asdict(self).items():
            object.__setattr__(self, key, _2float(val))

        
class Scenarios:

    def __init__(self,genTechs,storTechs, **options) -> None:

        self.genBase  = genTechs
        self.storBase = storTechs
    
        ScenarioCases = asdict(PossibleCases(**options))

        keys = [key for key in ScenarioCases]
        vals = [ScenarioCases[key] for key in keys]

        grid         = np.vstack([p.ravel() for p in np.meshgrid(*vals)]).T
        gridDF       = pd.DataFrame(grid,columns=keys)
        gridDF.index = [i for i in range(1,gridDF.shape[0]+1)]

        self.Cases = { 
            i : Case(**gridDF.loc[i,:].to_dict()) 
            for i in gridDF.index
        }
        self.CasesDF = gridDF
        
'''
NOTE FOR FUTURE: 
1. Need to update case-specific costs
2. Need to implement method for calculating case-specific battery starting hour
'''


    


#test = Scenarios(RenewInv=[0,0.25],CarbonTax=[0,100],LongInv=[0,None])
stop=True