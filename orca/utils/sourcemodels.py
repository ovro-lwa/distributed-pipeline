"""Just a bunch of dictionaries of source models.
"""
RFI_B = {
    "name": "Noise Power Lines2",
    "sys": "WGS84",
    "long": -118.3852914162684,
    "lat": 37.3078474772316,
    "el": 1214.248326037079,
    "rfi-frequencies": [2.60e7, 2.87e7, 3.13e7, 3.39e7, 3.65e7, 3.91e7, 4.18e7, 4.44e7, 4.70e7, 4.96e7,
                        5.22e7, 5.48e7, 5.75e7, 6.00e7, 6.27e7, 6.53e7, 6.79e7, 7.05e7, 7.32e7, 7.58e7,
                        7.84e7, 8.10e7],
    "rfi-I": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
              5.0, 5.0, 5.0]
}
CYG_A_UNPOLARIZED_RESOLVED = {
    "ref": "Michael (2016-07-30T10:52:45)",
    "name": "Cyg A",
    "components": [
        {
            "name": "1",
            "ra": "19h59m29.990s",
            "dec": "+40d43m57.53s",
            "I": 43170.55527073293,
            "Q": 0.0,
            "U": 0.0,
            "V": 0.0,
            "freq": 1.0e6,
            "index": [0.085, -0.178],
            "major-fwhm": 127.86780196141683,
            "minor-fwhm": 22.459884076169928,
            "position-angle": -74.50271323639498
        },
        {
            "name": "2",
            "ra": "19h59m24.316s",
            "dec": "+40d44m50.70s",
            "I": 6374.4647292670625,
            "Q": 0.0,
            "U": 0.0,
            "V": 0.0,
            "freq": 1.0e6,
            "index": [0.085, -0.178],
            "major-fwhm": 183.42701763410113,
            "minor-fwhm": 141.44188315233822,
            "position-angle": 43.449049376516
        }
    ]
}
CAS_A_UNPOLARIZED_RESOLVED = {
    "name": "Cas A",
    "components": [
        {
            "Q": 0.0,
            "minor-fwhm": 84.1,
            "V": 0.0,
            "major-fwhm": 208.89999999999998,
            "name": "1",
            "ra": "23h23m12.780s",
            "freq": 1.0e6,
            "index": [-0.77],
            "I": 205291.01635813876,
            "dec": "+58d50m41.00s",
            "U": 0.0,
            "position-angle": 38.9
        },
        {
            "Q": 0.0,
            "minor-fwhm": 121.9,
            "V": 0.0,
            "major-fwhm": 230.89999999999998,
            "name": "2",
            "ra": "23h23m28.090s",
            "freq": 1.0e6,
            "index": [-0.77],
            "I": 191558.43164385832,
            "dec": "+58d49m18.10s",
            "U": 0.0,
            "position-angle": 43.8
        },
        {
            "Q": 0.0,
            "minor-fwhm": 63.4649,
            "V": 0.0,
            "major-fwhm": 173.26,
            "name": "3",
            "ra": "23h23m20.880s",
            "freq": 1.0e6,
            "index": [-0.77],
            "I": 159054.81199800296,
            "dec": "+58d50m49.92s",
            "U": 0.0,
            "position-angle": 121.902
        }
    ]
}