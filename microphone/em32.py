import numpy as np

from .microphone import MicrophoneArray


class EigenmikeEM32(MicrophoneArray):
    '''
    Eigenmike em32 class that inherits from the MicrophoneArray class.
    '''

    def __init__(self):
        super(EigenmikeEM32, self).__init__(
            'Eigenmike 32', 'Rigid Spherical', 17.0, 'Omni')
        self._numelements = 32

        self._thetas = np.array([69.0, 90.0, 111.0, 90.0, 32.0, 55.0,
                                 90.0, 125.0, 148.0, 125.0, 90.0, 55.0, 21.0, 58.0,
                                 121.0, 159.0, 69.0, 90.0, 111.0, 90.0, 32.0, 55.0,
                                 90.0, 125.0, 148.0, 125.0, 90.0, 55.0, 21.0, 58.0,
                                 122.0, 159.0]) / 180.0 * np.pi

        self._phis = np.array([0.0, 32.0, 0.0, 328.0, 0.0, 45.0, 69.0, 45.0, 0.0, 315.0,
                               291.0, 315.0, 91.0, 90.0, 90.0, 89.0, 180.0, 212.0, 180.0, 148.0, 180.0,
                               225.0, 249.0, 225.0, 180.0, 135.0, 111.0, 135.0, 269.0, 270.0, 270.0,
                               271.0]) / 180.0 * np.pi

        self._radius = 4.2e-2

        self._weights = np.ones(32)

        self._info = 'Eigenmike em32 needs to be calibrated using the software tool provided mh Acoustics before use.'

    def returnAsStruct(self):
        '''
        Returns the attributes of the Eigenmike em32 as a struct

        :return: dict object with the name, type, thetas, phis, radius, weights, version, numelements, directivity, info fields
        '''
        em32 = {'name': self._micname,
                'type': self._arraytype,
                'thetas': self._thetas,
                'phis': self._phis,
                'radius': self._radius,
                'weights': self._weights,
                'version': self._ver,
                'numelements': self._numelements,
                'directivity': self._directivity,
                'info': self._info}
        return em32
