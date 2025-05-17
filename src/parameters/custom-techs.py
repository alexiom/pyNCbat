
# HOW TO USE: To include custom electricity technologies, you will need to know
#             whether it is a generation (produces electricity) or storage
#             (stores electricity that has already been generated) technology.
#             Depending on the type of technology, put the required paramters 
#             in the indicated sections of this script. See the section at the
#             end of this scrip title REQUIRED PARAMETERS for a list and 
#             description of the required parameters for each type of custom 
#             technologies. The default unit of electricty is a gigawatt (GW).

# =========================================================================== #
#                PUT PARAMETERS FOR CUSTOM TECHNOLOGIES HERE                  #
# =========================================================================== #

# NOTE: In all the formats for naming the custom objects below, [NAME-ABBREV] 
#       should be replacted with an abbreviation for name of 
#       the custom technology in all caps & without spaces, hyphens, 
#       underscores and brackets.


#### ---- GENERATION TECHNOLOGIES

### RENTAL/CAPITAL COSTS FOR CUSTOM ELECTRICITY GENERATION TECHNOLOGIES:
#   - NOTE: Object name format: RENT_[NAME-ABBREV]


### MARGINAL COSTS FOR CUSTOM ELECTRICITY GENERATION TECHNOLOGIES:
#   - NOTE: Object name format: MC_[NAME-ABBREV]


### CO2 EMISSIONS FOR CUSTOM ELECTRICITY GENERATION TECHNOLOGIES:
#   - NOTE: Object name format: CO2_[NAME-ABBREV]



#### ---- STORAGE TECHNOLOGIES

### RENTAL/CAPITAL COSTS FOR CUSTOM ELECTRICITY STORAGE TECHNOLOGIES:
#   - NOTE: Object name format: RENT_[NAME-ABBREV]


### MARGINAL COSTS FOR CUSTOM ELECTRICITY STORAGE TECHNOLOGIES:
#   - NOTE: Object name format: MC_[NAME-ABBREV]


### SELF DISCHARGE RATE (ALPHA) FOR CUSTOM ELECTRICITY STORAGE TECHNOLOGIES:
#   - NOTE: Object name format: ALPHA_[NAME-ABBREV]


### ROUND TRIP EFFECIENCY (RTE) FOR CUSTOM ELECTRICITY STORAGE TECHNOLOGIES:
#   - NOTE: Object name format: RTE_[NAME-ABBREV]



# =========================================================================== #
#                            REQUIRED PARAMETERS                              #
# =========================================================================== #

# Generation Technology
#   1. Rental/Capital Cost: The yearly fixed cost of implementing the tech.
#   2. Marginal Cost: The (constant) cost of producing another unit  of 
#        electricity. For renewable techs, set to 0.0.
#   3. CO2 Emissions: Carbon dioxide emissions from producing one unit of 
#        electricity. If renewable/clean, set to 0.0.

# Storage Technology
#   1. Rental/Capital Cost: The yearly fixed cost of implementing the tech.
#   2. Marginal Cost: The (constant) cost of storing and releasing one unit of 
#        electricity. Should almost always be 0.0.
#   3. Self Discharge Rate (alpha): The hourly self dischage rate of the tech, 
#        denoted as alpha i.e. the share of electricity lost every hour due to 
#        the tech self discharging.
#   4. Round Trip Efficiency (RTE): The share of remaining electricity after 
#        transmission losses FROM entering and immediately leaving storage. 
#        The RTE will be used to obtain the tramission loss rate, delta, using 
#        the following formula: delta = (1-RTE)/(1+RTE)