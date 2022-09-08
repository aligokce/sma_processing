'''
    Referenced from higrid.Microphone
'''

import numpy as np

class Microphone(object):
    '''
    Microphone class
    '''

    def __init__(self, name='Generic', version='1.0', direct='Omnidirectional'):
        '''
        Constructor

        :param name: Name of the microphone
        :param version: Version of the microphone
        :param direct: Directivity of the microphone (str)
        '''
        self._micname = name
        self._ver = version
        self._directivity = direct

    def getname(self):
        '''
        Getter for the name

        :return: Name (str) of the microphone object
        '''
        return self._micname

    def getversion(self):
        '''
        Getter for the version

        :return: Version (str) of the microphone object
        '''
        print(self._ver)

    def setname(self, name):
        '''
        Setter for the name attribute

        :param name: Name of the microphone (str)
        '''
        self._micname = name

    def setversion(self, version):
        '''
        Setter for the version attribute

        :param name: Version of the microphone (str)
        '''
        self._ver = version


class MicrophoneArray(Microphone):
    '''
    MicrophoneArray Class that inherits from the Microphone class
    '''

    def __init__(self, name, typ, version, direct):
        '''
        Constructor

        :param name: Name of the array
        :param typ: Type of the array (e.g. 'RSMA', 'OSMA', 'Linear')
        :param version: Version of the array (e.g. '1.0a')
        :param direct: Directivity of components (e.g. 'Onmidirectional')
        '''
        super(MicrophoneArray, self).__init__(name, version, direct)
        self._arraytype = None
        self.__arraytype = typ

    def gettype(self):
        '''
        Getter for array type

        :return: Type of the array (str)
        '''
        return self.__arraytype

    def settype(self, typ):
        '''
        Setter for array type

        :param typ: Type of the array (str)
        '''
        self.__arraytype = typ