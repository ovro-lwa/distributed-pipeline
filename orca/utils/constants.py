from astropy import units as u

# From Google Map cutout on http://www.tauceti.caltech.edu/LWA/
LWA_LAT = 37.239883 * u.deg
LWA_LON = -118.281675 * u.deg
LWA_ALTITUDE = 124 * u.m

# Could also write a method to figure these out from an ms post-expansion
LWA_N_CHAN = 109
LWA_N_SPW = 22
LWA_N_ANT = 256
