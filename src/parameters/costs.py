# Capital Costs of Electrictiy Generation/Storage Technologies
# - Units: $ per MW
RENT_SUN   = 83274.0             # solar
RENT_WIND  = 132602.0            # wind
RENT_NUKE  = 528307.0            # nuclear
RENT_GASCC = 79489.0             # gas combined cycle
RENT_GASP  = 54741.0             # gas peak
RENT_SHORT = 75739.0 / 4         # short duration storage
RENT_LONG  = RENT_SHORT * 1.885  # long duration storage
#RENT_BAT   = 75739.0 / 4         # batteries

# Marginal Costs of Generating Electricity
# - Units: $ per MWh
MC_SUN   = 0.0   # solar (renewable)
MC_WIND  = 0.0   # wind (renewable)
MC_NUKE  = 2.38  # nuclear
MC_GASCC = 26.68 # gas combined cycle
MC_GASP  = 44.13 # gas peak
MC_SHORT = 0.0   # short duration storage (free to operate)
MC_LONG  = 0.0   # long duration storage (could have MC in future)
#MC_BAT   = 0.0   # batteries (free to operate)

rentDict = {key.replace('RENT_','') : val for key, val in globals().items() if key.startswith('RENT_')}
rentDict = {key.strip().lower() : val for key, val in rentDict.items()}
rentDict = {key.replace('gascc','gasCC').replace('gasp','gasP') : val for key, val in rentDict.items()}

mcDict = {key.replace('MC_','') : val for key, val in globals().items() if key.startswith('MC_')}
mcDict = {key.strip().lower() : val for key, val in mcDict.items()}
mcDict = {key.replace('gascc','gasCC').replace('gasp','gasP') : val for key, val in mcDict.items()}