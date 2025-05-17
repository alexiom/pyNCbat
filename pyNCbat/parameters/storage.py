# =========================================================================== #
#                            STORAGE PARAMETERS                               #
# =========================================================================== #

ALPHA_SHORT = 0.9999997
RTE_SHORT   = 0.93

ALPHA_LONG = 1    # self discharge rate
RTE_LONG   = 0.83 # round trip effeciency

#ALPHA_BAT = 0.9999997
#RTE_BAT   = 0.8

#ALPHA_LITHION = 0.9999997
#RTE_LITHION = 0.83

#ALPHA_PUMP = 1
# #RTE_PUMP = 0.8

alphas = {key.replace('ALPHA_','') : val for key, val in globals().items() if key.startswith('ALPHA_')}
alphas = {key.strip().lower() : val for key, val in alphas.items()}

rtes = {key.replace('RTE_','') : val for key, val in globals().items() if key.startswith('RTE_')}
rtes = {key.strip().lower() : val for key, val in rtes.items()}

deltas = {key : (1-val)/(1+val) for key, val in rtes.items()}

