CO2_SUN   = 0.0
CO2_WIND  = 0.0
CO2_NUKE  = 0.0
CO2_GASCC = 0.74529/2.205
CO2_GASP  = 1.15889/2.205


co2s = {key.replace('CO2_','') : val for key, val in globals().items() if key.startswith('CO2_')}
co2s = {key.strip().lower() : val for key, val in co2s.items()}
co2s = {key.replace('gascc','gasCC').replace('gasp','gasP') : val for key, val in co2s.items()}