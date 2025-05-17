from parameters.costs import rents, mcs
from parameters.storage import alphas, deltas
from parameters.generation import co2s
from classes import GenTech, StorTech

storages = tuple(key for key in alphas.keys())
techs    = tuple(key for key in rents.keys())
gentechs = tuple(key for key in techs if key not in storages)

storTechs_base = {
    key : StorTech(capcost=rents[key],alpha=alphas[key],delta=deltas[key]) 
    for key in storages
}

genTechs_base = {
    key : GenTech(margcost=mcs[key],capcost=rents[key],co2=co2s[key])
    for key in storages
}



stop=True