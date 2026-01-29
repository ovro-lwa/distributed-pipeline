"""Physical and observational constants for OVRO-LWA.

Contains telescope configuration constants including channel counts,
spectral window counts, and antenna counts.

Note:
    These are legacy values for the original 256-antenna array.
    The expanded array uses telescope configuration from configmanager.
"""# Could also write a method to figure these out from an ms post-expansion
LWA_N_CHAN = 109
LWA_N_SPW = 22
LWA_N_ANT = 256
