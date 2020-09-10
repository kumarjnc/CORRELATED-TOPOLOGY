hbar = 1.054571628E-34; # In J.s - from NIST (2006)
hconst = 6.62606896E-34; # In J.s - from NIST (2006)
a0 = 5.291772108E-11; # m - from NIST (2010)
Eh = 4.35974394E-18; # J - from NIST (2006)
clight = 299792458; # m/s - from NIST (2006)
clightau = 137.0359996287515 # in au
eps0 = 8.8541878176E-12; # F/m - from NIST (2006)
electronmass = 9.10938215E-31 # kg - from NIST (2006)
eV = 1.602176487e-19 # in J - from NIST (2006)
AvagadrosNumber = 6.02214179E23;
He4molarmass = 4.002603E-3; # in kg/mol from NIST (2010)
__He4mass = He4molarmass/AvagadrosNumber;
__He4redmass = __He4mass/2; # for a diatomic He pair
boltzmannk = 1.38e-23 # J/K - from Halliday, Resnick, Walker
echarge = 1.602176487e-19 # C - from wikipedia
amu = 1.660538782e-27 # kg - from NIST (2006)

# For working in au
# PJ value is the nuclear mass for the Przybytek and Jeziorski potential
# TODO: Gotta make my code use redmassPJ now instead of just redmass
He4redmass = __He4redmass / electronmass;
He4redmassPJ = ( (__He4mass / electronmass) - 2 ) / 2;

He3molarmass = 3.016029E-3; # in kg/mol from NIST (2006)
__He3mass = He3molarmass/AvagadrosNumber;
__He3redmass = __He3mass/2; # for a diatomic He pair
He3redmass = __He3redmass / electronmass;
He3redmassPJ = ( (__He3mass / electronmass) - 2 ) / 2;

He34redmass = (__He3mass * __He4mass) / (__He3mass + __He4mass) / electronmass

Eh_to_MHz = Eh / hconst / 10**6
auI_to_Wcm2 = 1 / (hbar / Eh**2 * 100**2 * a0**2)
