#import sys
#sys.path.append('~/xiaoni/MCD_DA-master/classification/model')
from usps import*
import usps
#from .usps import AdversarialNetwork 
#from .usps import Predictory 
#from .usps import Mine_1


import svhn2mnist
#from .usps import AdversarialNetwork 

from .syn2gtrsb import* 
#from .syndig2svhn import* 

def discriminator_mi(source, target):
    if source == 'usps' or target == 'usps':
        return usps.Domain_discriminator(48*4*4)
    if source == 'svhn':
        return svhn2mnist.Domain_discriminator(3072)
    if source == 'synth':
        return syn2gtrsb.Domain_discriminator( )


def discriminate_Type(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return usps.discriminate_Type_mi()
    elif source == 'svhn':
        return svhn2mnist.discriminate_Type_mi()
    elif source == 'synth':
        return syn2gtrsb.discriminate_Type_mi()

def Generator(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return usps.Feature()
    elif source == 'svhn':
        return svhn2mnist.Feature()
    elif source == 'synth':
        return syn2gtrsb.Feature()

def Classifier(source, target):
    if source == 'usps' or target == 'usps':
        return usps.Predictor()
    if source == 'svhn':
        return svhn2mnist.Predictor()
    if source == 'synth':
        return syn2gtrsb.Predictor()

def Classifier_y(source, target):
    if source == 'usps' or target == 'usps':
        return usps.Predictory()
    if source == 'svhn':
        return svhn2mnist.Predictor()
    if source == 'synth':
        return syn2gtrsb.Predictor()

def discriminator(source, target):
    if source == 'usps' or target == 'usps':
        return usps.AdversarialNetwork(48*4*4)
    if source == 'svhn':
        return svhn2mnist.AdversarialNetwork(3072)
    if source == 'synth':
        return syn2gtrsb.AdversarialNetwork( )




