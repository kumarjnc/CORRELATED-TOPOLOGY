#This folder  contains script file and DMFT+CTAUX code along with codes which calculates 
the QSH state based on toplogical Hamiltonian approximation (PHYS. REV. X 2, 031008 (2012)) 
and spin Chern number calculation (Fukui et al PHYSICAL REVIEW B 75, 121403 (R) 2007).

1. CTAUX : It contains the solver code CTQMC (CTAUX) for spin-mixed (zero spin mixing) case.

2. DMFT :  This contains DMFT routine (RDMFT_TISM.py is the main function) is real space version which can Handle staggering, trap and disorder.it also contains the script to extract the self-energy (selfenergy_extraction.py) at zero (infinite) frequency, which is needed for QSH calculations. Additionaly contains the scripts to calculate the staggering magnetization (staggering_magnetization.py), staggering occupancy (staggering_uccupancy.py) and double occupancy (double_occupancy.py). 

3. SCRIP_TO_RUN: This folder contains the shell script where various input specification can be made. User can modify the command for storage according to comfort.

4.QSH_TOPOLOGICAL_HAMILTONIAN_CALCULATION : This folder contains the code which calculates the Z_2 invariant (0 for trivial and 1 for QSH state) using topological Hamiltonian concept. It uses the self-energy output of the DMFT.

5.QSH_edge state calculation: This folder contains the code to calculate the edge states. This is benchmarked by producung the results from Goldman etal paper (PRL 105, 255302 (2010)). It works for spin mixed and zero spin mixed case. The magnetic flux strength \alpha =p/q can be choosen as well. 
