from . import encodings

class PoissonEncoder:
    r"""Encode a set of spiking intensities to spike trains using a Poisson distribution.
    
    Adapted from:
        https://github.com/Hananel-Hazan/bindsnet/blob/master/bindsnet/encoding/encodings.py

    Creates a callable PoissonEncoder which encodes as defined in ``bindsnet.encoding.poisson``

    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    """

    def __init__(self, duration, dt):
        self.duration = duration
        self.dt = dt

    def __call__(self, intensities):
        return poisson_encoding(intensities, self.duration, self.dt)