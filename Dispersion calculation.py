# -*- coding: utf-8 -*-
"""
Code made by Pedro Sansoldo from the University of Adelaide's Photonic Integrated Chips (PIC) Group
PIC Group's Github repository: https://github.com/psansoldo-au/PIC-Group-Adelaide-University/
Numerical differentiation version using central difference to compute beta''(omega)
with corrected finite difference formula for non-uniform omega spacing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Constants
c = 299792458  # speed of light in m/s

# Filenames and labels
files_and_labels = [
     ("wd600 Dp v3 com40 AlGaAs SiO2.txt", "wd600"),
     ("wd700 Dp v3 com40 AlGaAs SiO2.txt", "wd700"),
     ("wd800 Dp v3 com40 AlGaAs SiO2.txt", "wd800"),
]

for filename, label in files_and_labels:
    
    ###########################################################################
    
    # Variables created from comsol
    
    data = pd.read_csv(filename, delim_whitespace=True, skiprows=5)
    
    wlp = data.iloc[:, 1].to_numpy() * 1e-9  # Convert nm to m
    exec(f"wlp_{label} = wlp")
    
    rneff = data.iloc[:, 5].to_numpy() # [1]
    exec(f"rneff_{label} = rneff")
    
    def omp_f(wlp):
        omp = 2 * np.pi * c / wlp
        return omp
    
    omp = omp_f(wlp)
    exec(f"omp_{label} = omp")
    
    def beta_f(rneff,wlp):
        beta = 2 * np.pi * rneff / wlp
        return beta
    
    beta = beta_f(rneff,wlp)
    exec(f"beta_{label} = beta")
    
    ###########################################################################
    
    # First derivatives
    
    def d1_py_f(x, y):
        d1_py = np.gradient(y, x)
        return d1_py

    d1rneff_wlp_py = d1_py_f(wlp, rneff)
    exec(f"d1rneff_wlp_py_{label} = d1rneff_wlp_py")

    ###########################################################################
    
    # Second order derivative

    def d2_py_f(x, y):
        d2_py = np.gradient(np.gradient(y, x), x)
        return d2_py
    
    d2rneff_wlp_py = d2_py_f(wlp, rneff)
    exec(f"d2rneff_wlp_py_{label} = d2rneff_wlp_py")
        
    ###########################################################################
    
    # Group index (ng)
    
    def ng_rneff_f(rneff, wlp, d1rneff_wlp):
        ng_rneff = rneff - wlp * d1rneff_wlp
        return ng_rneff

    ng_rneff_py = ng_rneff_f(rneff, wlp, d1rneff_wlp_py)
    exec(f"ng_rneff_py_{label} = ng_rneff_py")
    
    ###########################################################################
    
    # D1 (FSR)
    
    def D1_f(ng, r_mrr):
        c = 299792458  # speed of light in m/s
        D1 = c / (ng * 2 * np.pi * r_mrr)
        return D1
    
    r_mrr = 50e-6

    D1_py = D1_f(ng_rneff_py,r_mrr) # D1 [Hz]
    exec(f"D1_py_{label} = D1_py")

    # D2 Mode dispersion
    
    def D2_f(wlp, D1, d1D1_wlp):
        c = 299792458  # speed of light in m/s
        D2 = - (wlp**2 / (2 * np.pi * c)) * D1 * d1D1_wlp
        return D2

    D2_py = D2_f(wlp, D1_py, d1_py_f(wlp,D1_py)) / (2 * np.pi) # D2 [Hz]
    exec(f"D2_py_{label} = D2_py")
    
    ###########################################################################
    
    # Disp Dispersion parameter, GVD
    
    def Disp_f(wlp, rneff):
        c = 299792458  # c [m/s] Speed of light
        Disp= - (wlp / c) * np.gradient(np.gradient(rneff, wlp), wlp)
        return Disp
    
    Disp = Disp_f(wlp,rneff) * 1e6 # Disp [ps * (nm * km )e-1]
    exec(f"Disp_{label} = Disp")
    
    print(f"{label}: Disp[0] = {Disp[0]:.2f}, min = {Disp.min():.2f}, max = {Disp.max():.2f}")

    
    ###########################################################################
    
    ## Plotting
    
    # omp
    
    plt.figure(1)
    plt.plot(wlp * 1e9, omp, label=label, linewidth=2)
    
    # rneff
    
    plt.figure(2)
    plt.plot(wlp * 1e9, rneff, label=label, linewidth=2)
    
    # beta
    
    plt.figure(3)
    plt.plot(wlp * 1e9, beta, label=label, linewidth=2)
    
    # d1rneff
    
    plt.figure(6)
    plt.plot(wlp * 1e9, d1rneff_wlp_py, label=label, linewidth=2)
    
    # d2rneff
    
    plt.figure(9) 
    plt.plot(wlp * 1e9, d2rneff_wlp_py, label=label, linewidth=2)
    
    # ng
    
    plt.figure(12)
    plt.plot(wlp * 1e9, ng_rneff_py, label=label, linewidth=2)
 
    # D1
    
    plt.figure(15)
    plt.plot(wlp * 1e9, D1_py, label=label, linewidth=2)
    
    # D2

    plt.figure(18)
    plt.plot(wlp * 1e9, D2_py * 1e-6 , label=label, linewidth=2)
    
    # Disp

    plt.figure(19)
    plt.plot(wlp * 1e6, Disp, label=label, linewidth=2) # Disp [ps * nm 1e-1 * km 1e-1]

    # # GVD

    # plt.figure(20)
    # plt.plot(wlp * 1e6, GVD, label=label, linewidth=2) # Disp [ps * nm 1e-1 * km 1e-1]
    
    # Save D2 data to file
    D2_py_Hz = D2_py * (2 * np.pi)  # Convert to Hz
    
    output_array = np.column_stack((wlp * 1e6, D2_py_Hz))  # Wavelength in µm
    output_filename = f"{label}_D2_output.txt"
    np.savetxt(output_filename, output_array, fmt="%.3f %.8e")

# omp

plt.figure(1)
plt.title("omp")
plt.xlabel("Wavelength [nm]")
plt.grid(True)
plt.legend()
plt.tight_layout()

# rneff

plt.figure(2)
plt.title("rneff")
plt.xlabel("Wavelength [nm]")
plt.grid(True)
plt.legend()
plt.tight_layout()

# beta

plt.figure(3)
plt.title("beta")
plt.xlabel("Wavelength [nm]")
plt.grid(True)
plt.legend()
plt.tight_layout()

# d1beta

plt.figure(6)
plt.title("d1rneff_wlp_py")
plt.xlabel("Wavelength [nm]")
plt.grid(True)
plt.legend()
plt.tight_layout()

# d2beta

plt.figure(9)
plt.title("d2rneff_wlp_py")
plt.xlabel("Wavelength [nm]")
plt.grid(True)
plt.legend()
plt.tight_layout()

# ng

plt.figure(12)
plt.title("ng_rneff_py")
plt.xlabel("Wavelength [nm]")
plt.grid(True)
plt.legend()
plt.tight_layout()

# D1

plt.figure(15)
plt.title("D1_py")
plt.xlabel("Wavelength [nm]")
plt.grid(True)
plt.legend()
plt.tight_layout()

# D2

plt.figure(18)
plt.title("D2_py")
plt.xlabel("Wavelength [nm]")
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.xlim([800, 2300])
#plt.ylim([-50, 100])

# Disp

plt.figure(19)
plt.title("Dispersion parameter (D) [ps/(nm*km)]")
plt.xlabel("Wavelength [um]")
plt.ylabel("D [ps/(nm*km)]")
plt.axhline(y=0, color='blue', linestyle='--')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xlim([1.3, 1.7])
plt.ylim([-500, 1500])
# ax = plt.gca()  # get current axis
# ax.xaxis.set_major_locator(MultipleLocator(0.1)) 
#plt.ylim([-50, 100])

# # GVD

# plt.figure(20)
# plt.title("GVD (D) [ps/(nm*km)]")
# plt.xlabel("Wavelength [um]")
# plt.ylabel("GVD [ps/(nm*km)]")
# plt.axhline(y=0, color='blue', linestyle='--')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.xlim([1.3, 1.7])
# # plt.ylim([-500, 1500])

plt.show()
