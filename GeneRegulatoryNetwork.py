#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:26:22 2020

@author: andrewg
"""
import numpy as np
from scipy.integrate import odeint
from Global_Constant import time_hours, diff_rate_hours
import time

idx={
    "PtcI": 0,
    "Ptc":  1,
    "GliF": 2,
    "GliA": 3,
    "GliR": 4,
    "Pax":  5,
    "Oli":  6,
    "Nkx":  7,
    "Irx":  8
}
nSpecies = len(idx)

def Hill (x):
    return x / ( 1. + x )

f_A = 10.
c_GliA = 10.

def shh_gli_poni (X, t, shh,
      # Shh-Gli
      K_Pol_Ptc=1.59687005e+00,
      K_Pol_Gli=4.71752834e+01,
      K_Gli_Ptc=8.99478929e-01,
      Km_Ptc_Gli=3.38813445e+02,
      w_Ptc_A=1.31959134e-01,
      delta_PtcI=3.60923086e-02,
      alpha_Ptc=1.31959134e-01,
      alpha_GliF=9.00660795e-01,
      w_Gli_A=1.56277655e+00,
      w_Gli_R=2.10469474e-01,
      delta_Ptc=3.60923086e-02,
      delta_GliF=6.01546585e-01,
      delta_GliA=3.53728598e-01,
      delta_GliR=1.96965306e-01,
      B_Shh_Ptc=1.37991521e+02,
      # PONI
      alpha_Pax=2.00000000e+00,
      alpha_Oli=2.00000000e+00,
      alpha_Nkx=2.00000000e+00,
      alpha_Irx=2.00000000e+00,
      delta=2.00000000e+00,
      K_Pol_Pax=1.70887077e+00,
      K_Oli_Pax=1.19444166e+00,
      K_Nkx_Pax=2.52956598e+01,
      K_Pol_Oli=8.42805553e+01,
      K_Gli_Oli=3.50339091e+01,
      K_Nkx_Oli=5.35565407e+01,
      K_Irx_Oli=3.41567284e+01,
      K_Pol_Nkx=2.12589420e+01,
      K_Gli_Nkx=2.99506875e+02,
      K_Pax_Nkx=5.76386046e+00,
      K_Oli_Nkx=1.79875234e+01,
      K_Irx_Nkx=6.54064389e+01,
      K_Pol_Irx=2.02603123e+01,
      K_Oli_Irx=5.95046336e+01,
      K_Nkx_Irx=4.06288460e+01,
      # Feedback
      K_Nkx_Gli=9.23547567e+01,
      f_Nkx_Gli=1.77437736e-01,
      K_Oli_Gli=3.49396264e+00,
      f_Oli_Gli=6.14619675e-02
     ):
    '''
    Dynamical system of the full model
    '''
    
    ptcI, ptc, glif, glia, glir, pax, oli, nkx, irx = X
    
    # feedback weights
    aux_1 = 1 + f_Oli_Gli * K_Oli_Gli * oli
    aux_1 *= 1 + f_Nkx_Gli * K_Nkx_Gli * nkx
    aux_2 = 1 + K_Oli_Gli * oli
    aux_2 *= 1 + K_Nkx_Gli * nkx

    feedbackPtc = 1.
    feedbackGli = (aux_1/aux_2)*(aux_1/aux_2)
    
    # intermediate Ptc
    aux_1 = 1. + c_GliA * K_Gli_Ptc * glia
    aux_2 = 1. + K_Gli_Ptc*(glia + glir)
    aux_3 = aux_1*aux_1 / ( aux_2*aux_2 ) * feedbackPtc
    f_ptcI = alpha_Ptc * Hill( K_Pol_Ptc * aux_3 )
    f_ptcI += - ( delta_PtcI + w_Ptc_A ) * ptcI

    # Ptch
    f_ptc = w_Ptc_A * ptcI 
    f_ptc += - ( delta_Ptc + B_Shh_Ptc * shh ) * ptc

    # GliFL
    aux_1 = feedbackGli
    f_glif = alpha_GliF * Hill( K_Pol_Gli * aux_1 )
    f_glif += - ( w_Gli_A / ( 1. + Km_Ptc_Gli * ptc ) + w_Gli_R + delta_GliF ) * glif

    # GliA
    f_glia = w_Gli_A / ( 1. + Km_Ptc_Gli * ptc ) * glif
    f_glia += - delta_GliA * glia

    # GliR
    f_glir = w_Gli_R * glif
    f_glir += - delta_GliR * glir

    # Pax
    # Repression by Olig
    aux_1 = 1./(1. + K_Oli_Pax * oli)
    aux_1 *= aux_1
    # Repression by Nkx
    aux_2 = 1./(1. + K_Nkx_Pax * nkx)
    aux_2 *= aux_2
    f_pax = alpha_Pax * Hill(K_Pol_Pax  * aux_1 * aux_2)
    f_pax += - delta * pax

    # Olig
    # Activation by Gli
    aux_1 = 1. + f_A * K_Gli_Oli * glia
    aux_1 /= 1. + K_Gli_Oli * ( glia + glir )
    # Repression by Nkx
    aux_2 = 1./(1. + K_Nkx_Oli * nkx)
    aux_2 *= aux_2
    # Repression by Irx
    aux_3 = 1./(1. + K_Irx_Oli * irx)
    aux_3 *= aux_3
    f_oli = alpha_Oli * Hill(K_Pol_Oli  * aux_1 * aux_2 * aux_3)
    f_oli += - delta * oli

    # Nkx
    # Activation by Gli
    aux_1 = 1. + f_A * K_Gli_Nkx * glia
    aux_1 /= 1. + K_Gli_Nkx * ( glia + glir )
    # Repression by Pax
    aux_2 = 1./(1. + K_Pax_Nkx * pax)
    aux_2 *= aux_2
    # Repression by Olig
    aux_3 = 1./(1. + K_Oli_Nkx * oli)
    aux_3 *= aux_3
    # Repression by Irx
    aux_4 = 1./(1. + K_Irx_Nkx * irx)
    aux_4 *= aux_4
    f_nkx = alpha_Nkx * Hill(K_Pol_Nkx  * aux_1 * aux_2 * aux_3 * aux_4)
    f_nkx += - delta * nkx

    # Irx
    # Repression by Olig
    aux_1 = 1./(1. + K_Oli_Irx * oli)
    aux_1 *= aux_1
    # Repression by Nkx
    aux_2 = 1./(1. + K_Nkx_Irx * nkx)
    aux_2 *= aux_2
    f_irx = alpha_Irx * Hill(K_Pol_Irx  * aux_1 * aux_2)
    f_irx += - delta * irx
        
    return np.array([f_ptcI, f_ptc, f_glif, f_glia, f_glir, f_pax, f_oli, f_nkx, f_irx])


class GRN(object):
    """
        GRN object

    """
    
    def __init__(self,state,GRN_function,sig_input,lost_morphogen=None):
        self.state=state #state[i] is the state of the GRN
        self.GRN_function=GRN_function #GRN function for all cells
        self.sig_input=sig_input
        self.lost_morphogen=np.zeros(len(state))

    @property
    def n_cells(self):
        return len(self.state)

    @property
    def diff_rates(self):
        rates = 0.05 * time_hours * diff_rate_hours * np.ones(self.n_cells)
        olig_high = np.where(self.state[:,idx['Oli']] > .7)[0]
        rates[olig_high] = 0.5 * time_hours * diff_rate_hours
        return rates
        
    def evolve(self,time,dt,sig_input,bind_rate): #could put self.sig_input...
        """
            One time step dt from time to time+dt with Euler method
        
        """
        # print(f'\nshape GRN state before = {self.state.shape}')

        self.lost_morphogen = bind_rate * sig_input * self.state[:,idx['Ptc']]
        f = np.array( list(map(self.GRN_function,self.state, np.repeat(time,self.n_cells), sig_input)) )
        self.state = self.state + dt * f
        # print(f'shape GRN state after = {self.state.shape}')
    
    def divide(self,ready,factor=.5):
        '''
            Define gene expression in daughter cells

        '''
        # print(f'cells before divisions: {self.n_cells}')
        # print(f'cells dividing: {len(ready)}')
        for k in ready:
            # gene expression in daughter is 'factor' times the one of the parent
            daughter_state = factor*self.state[k].copy() 
            old_state = self.state.copy()
            aux = np.vstack((old_state,daughter_state))
            self.state = np.vstack((aux, daughter_state))
        # print(f'cells after divisions: {self.n_cells}')

        
def build_GRN(n, GRN_function=shh_gli_poni, state=None , sig_input=None):
    """
        Function to initialize the GRN_basic object
        ..Why this was not included in the __init__ of the GRN, is a mystery..
    """
    if state is None:
        state= np.array([[1.,     1.,      1., .00001, 1.,
                          1., .00001, 0.00001,      1.]]*n)
    else: 
        state=state

    if sig_input is None:
        sig_input = np.zeros(n) #no signal to start
    else:
        sig_input = sig_input

    return GRN(state, GRN_function, sig_input) 
