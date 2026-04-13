# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 11:45:45 2026

@author: agilj
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import rotate
import time
from skimage.draw import polygon

from skimage.transform import resize

from ClassFile_OpticalSys import OpticalSys


"""
Compute diffraction-limited MTF for circular aperture.

Parameters:
    f: torch.Tensor (N,)
        The frequencies we are interested in.
    fc: float
        The diffraction-limited cutoff frequency.
        
Returns:
    torch.Tensor (N,): The diffraction-limited MTF.
"""
def getDiffMTF(f, fc):
    mtf = torch.zeros_like(f)
    inside = torch.abs(f) <= fc
    x = torch.abs(f[inside] / fc)
    mtf[inside] = (2/torch.pi) * (torch.arccos(x) - x * torch.sqrt(1 - x**2))
    return mtf

"""
Make a clear circular aperture in arbitrary units and return it.

Parameters:
    apDiv: int
        The number of pixels per side for the discretized aperture.
    device: str
        The device ('cpu' or 'cuda') that the model runs on.         
Returns:
    torch.Tensor (apDiv, apDiv): A discretized circular aperture in arbitrary
                                 units, formatted for easy input into the
                                 optical system.
"""
def circularAp(apDiv, device='cpu'):
    # Build grid.
    x = (torch.arange(apDiv).to(device) \
         - (apDiv-1)/2)
    X, Y = torch.meshgrid(x, x, indexing='ij')
    # Get radius of each point.
    rho = torch.sqrt(X**2 + Y**2)
    # Restrict to a circle.
    apMap = rho<=apDiv/2
    
    return apMap

"""
Make a polygon aperture in arbitrary units and return it.

Parameters:
    apDiv: int
        The number of pixels per side for the discretized aperture.
    numSides: int
        The number of sides the polygon should have.
    apRotation: float
        How many rads to rotate the aperture from its predefined orientation.
    device: str
        The device ('cpu' or 'cuda') that the model runs on. 
Returns:
    torch.Tensor (apDiv, apDiv): A discretized polygonal aperture in arbitrary
                                 units, formatted for easy input into the
                                 optical system.
"""
def polygonAp(apDiv, numSides, apRotation, device='cpu'):
    # Get polygon radius
    R = apDiv//2
    # Get polygon vertices.
    angles = np.linspace(0, 2 * torch.pi, numSides, 
                         endpoint=False) + apRotation
    x = apDiv//2 + R * np.cos(angles)
    y = apDiv//2 + R * np.sin(angles)
    vertices = np.vstack((x, y)).T
    
    # Get points contained by vertices.
    aperture = np.zeros((apDiv, apDiv), 
                        dtype=bool)
    rr, cc = polygon(vertices[:,1], vertices[:,0], 
                     shape=aperture.shape)  # y, x order
    aperture[rr, cc] = True
    aperture = aperture.reshape((apDiv, apDiv))
    apMap = torch.Tensor(aperture).to(device)
    
    return apMap

"""
Make a Bernoulli Random aperture in arbitrary units and return it.

Parameters:
    apDiv: int
        The number of pixels per side for the discretized aperture.
    numPixels: int
        The number of unit cells in the random aperture. Each unit cell will
        be randomly assigned to be 0 or 1.
    seed: int
        The random seed to use in generating the random aperture.
    device: str
        The device ('cpu' or 'cuda') that the model runs on.        
Returns:
    torch.Tensor (apDiv, apDiv): A discretized polygonal aperture in arbitrary
                                 units, formatted for easy input into the
                                 optical system.
"""
def randomAp(apDiv, numPixels, seed=0, device='cpu'):
    # BERNOULLI RANDOM APERTURE
    # Set seed.
    gen = torch.Generator()
    gen.manual_seed(0)
    # Make grid.
    x = (torch.arange(apDiv).to(device) \
         - (apDiv-1)/2)
    X, Y = torch.meshgrid(x, x, indexing='ij')
    rho = torch.sqrt(X**2 + Y**2) # Round Pupil function
    # Build randomly aperture.
    aperture = torch.bernoulli(0.5*torch.ones(numPixels**2), generator=gen)
    aperture = aperture.reshape((numPixels,numPixels)).to(device)
    aperture[aperture<0] = 0
    aperture[aperture>0] = 1
    ones = torch.ones((int(apDiv/numPixels), 
                       int(apDiv/numPixels))).to(device)
    aperture = torch.kron(aperture, ones)
    aperture = aperture * (rho<=apDiv/2) # Iris in front.
    
    apMap = torch.Tensor(aperture).to(device)
    
    return apMap

"""
Make an aperture that blocks part of the ray bundle.

Parameters:
    apDiv: int
        The number of pixels per side for the discretized aperture.
    rowStart: int
        The row at which the aperture "turns on" and begins blocking rays.
    angle: float
        The angle in radians to rotate the aperture about its center.
    device: str
        The device ('cpu' or 'cuda') that the model runs on.        
Returns:
    torch.Tensor (apDiv, apDiv): A discretized blocking aperture in arbitrary
                                 units, formatted for easy input into the
                                 optical system.
"""
def blockingAp(apDiv, rowStart, angle, device='cpu'):
    # We make an array where the first M rows are clear and the remaining rows
    # are opaque. We then rotate this by some number of radians.
    # Define aperture
    x = np.zeros((apDiv,apDiv))
    # Block some rows
    x[rowStart:,:] = 1
    # Rotate
    apMap = rotate(x, angle*180/np.pi, reshape=False, order=1, mode='nearest')

    return apMap

#%%
if __name__ == '__main__':
    # Ray tracing parameters. Note that sparseASM requires a far greater
    # gridDiv value than fraunhofer. Fraunhofer often works with gridDiv=256
    # whereas sparseASM often requires gridDiv>4096 for accurate results.
    numObsPts = 2048
    obsSpacing = 1.5e-4 # mm
    apDiv = 256 # Aperture point divisions used for picking rays.
    fieldAngle_deg = 1 # Input field angle.
    fieldDir = 'y' # Axis forming fieldAngle_deg: 'x', 'y', or 'xy'.
    defocus = 0 # mm
    gridDiv = 256 # Pupil grid divisions used in propagation.
    mode = 'fraunhofer' # fraunhofer or sparseASM

    # Lens parameters.
    # We define the first element to be the system's front aperture
        # Surface curvature
    curvList = [np.inf, 33.55, -27.05, -125.6, -48.8, 59, np.inf] # mm
        # Distance to next surface
    thickList = [0, 7.5, 1.8, 10, 2, 3.4, 0]
        # Material surface is made of.
    matList = ['AIR', 'N-LAK22','N-SF6HT', 'AIR', 'N-BAF10', 'N-SF6HT', 'AIR']
        # Diameter of surface.
    diamList = [12.7, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4]
    # # Here's a random aperture case with part of the beam
    # # blocked.
    # apList = [randomAp(apDiv, 128, device='cpu'),
    #           None,
    #           None,
    #           blockingAp(apDiv, 60, -3*np.pi/16, device='cpu'),
    #           None,
    #           None,
    #           None]
    # Here's a clean case with a pentagon aperture.
    apList = [polygonAp(apDiv, 5, 0, device='cpu'),
              None,
              None,
              None,
              None,
              None,
              None]
        # Operating wavelength
    # wvlList = [0.55e-3, 0.45e-3, 0.5e-3, 0.6e-3, 0.65e-3, 0.7e-3] # mm
    wvlList = [0.8e-3] # mm
    QE = [1] # This is a per-spectrum weighting used to simulate QE effects.
    
    plt.figure(dpi=300)
    plt.imshow(apList[0],extent=[-diamList[0], diamList[0],
                                 -diamList[0], diamList[0]])
    plt.xlabel('X-Position (mm)')
    plt.ylabel('Y-Position (mm)')
    plt.title('System Front Aperture')
    plt.show()
    
    # Define optical system.
    sys = OpticalSys(curvList, thickList, matList, diamList, apList, 
                     device='cpu')
    
    #%% Compute PSF 
    # Get the ray input direction.
    fieldAngle = fieldAngle_deg * (torch.pi/180)
    c = np.cos(fieldAngle)
    s = np.sin(fieldAngle)
    # Angle wrt x-axis
    if fieldDir == 'x':
        inputDir = torch.Tensor([s,
                                 0,
                                 c])
    # Angle wrt y-axis
    if fieldDir == 'y':
        inputDir = torch.Tensor([0,
                                 s, 
                                 c])
    # Angle wrt the diagonal y=x
    if fieldDir == 'xy':
        inputDir = torch.Tensor([s/np.sqrt(2),
                                 s/np.sqrt(2),
                                 c])
        
    t1 = time.time()
    psfList, \
    mtfList, mtf_xAxis, \
    encircList, ensqList, radii = sys.computePSF(wvlList, 
                                                 inputDir, 
                                                 numObsPts, 
                                                 obsSpacing,
                                                 gridDiv=gridDiv, 
                                                 defocus=defocus, 
                                                 mode=mode)
    t2 = time.time()
    print('Time to get PSF: '+str(t2-t1)+' seconds.')
    
    #%% Plot PSF
    wvl = wvlList[0]
    psf = torch.stack(psfList)#.sum(dim=0)
    psf = psf * torch.Tensor(QE)[:,None,None]
    psf = psf.sum(dim=0)
    mtf = mtfList[0] # Just take first (primary) wavelength.
    encirc = encircList[0]
    ensq = ensqList[0]
    
    # Look at how the PSF looks on a specific focal plane.
    psfSize = psf.size()[0]
    # Get pixel size ratio.
    pxSize = 2 # microns
    pxSizeRatio = pxSize / (obsSpacing*1000)
    # Build interp function
    newSize = int(psfSize / (pxSize/(obsSpacing*1000)))
    psf_rs = resize(np.asarray(psf), (newSize,newSize))
    numPxToShow = newSize
    # Crop down to managable size
    psf_px = psf_rs[newSize//2-numPxToShow//2:newSize//2+numPxToShow//2+1,
                    newSize//2-numPxToShow//2:newSize//2+numPxToShow//2+1]
    
    plt.figure(dpi=300)
    plt.imshow(psf_px)
    plt.title('Predicted PSF (Center '+str(numPxToShow)+'x'+str(numPxToShow)+\
              ' px)')
    
    # Now we plot the MTF, PSF, energies, and x-y lineouts.
    obs_range = numObsPts*obsSpacing
    c = psf.size()[0]//2
    crop_range = 30 # microns
    crop_half = int(crop_range / (obsSpacing*1000))
    psf_region = psf[c-crop_half:c+crop_half,c-crop_half:c+crop_half]
    radMax = 2*crop_range / 1000
    
    # Format the subplot figure.
    fig, ax = plt.subplots(2, 2, dpi=300)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, 
                        hspace=0.6, wspace=0.4)
    matplotlib.rc('xtick', labelsize=8) 
    matplotlib.rc('ytick', labelsize=8) 
    
    # Plot PSF
    im1 = ax[0,0].imshow(psf_region / psf_region.max(),
              extent=[-crop_range, crop_range,
                      -crop_range, crop_range],
              vmin=0, vmax=1)    
    ax[0,0].set_xlabel('X-Position ($\mu$m)', fontsize=10)
    ax[0,0].set_ylabel('Y-Position ($\mu$m)', fontsize=10)
    ax[0,0].set_title('PSF: '+mode)
    ax[0,0].set_xticks([-crop_range,0,crop_range])
    ax[0,0].set_yticks([-crop_range,0,crop_range])
    ax[0,0].set_xticklabels([-crop_range,0,crop_range], fontsize=8)
    ax[0,0].set_yticklabels([-crop_range,0,crop_range], fontsize=8)
    cbar = fig.colorbar(im1, ax=ax[0,0])
    cbar.ax.tick_params(labelsize=8)
    
    # Get diffraction MTF curve.
    NA = torch.sin(torch.arctan((diamList[0]/2)/sys.efl)).cpu()
    cutoffFreq = 2*NA / wvl
    diffMTF = getDiffMTF(mtf_xAxis, cutoffFreq)
    
    # Plot MTF
    c = mtf.size()[0]//2
    ax[1,0].plot(mtf_xAxis[c:c+150], mtf[c:c+150])
    ax[1,0].plot(mtf_xAxis[c:c+150], diffMTF[c:c+150])
    ax[1,0].set_xlabel('Spatial Frequency (lp/mm)', fontsize=10)
    ax[1,0].set_ylabel('Modulation', fontsize=8)
    ax[1,0].set_title('System MTF: Y-Projection')
    ax[1,0].axis([0,100,0,1])
    ax[1,0].set_xticks(np.arange(6)*20)
    ax[1,0].set_yticks(np.arange(6)/5)
    ax[1,0].set_xticklabels(np.arange(6)*20, fontsize=8)
    ax[1,0].set_yticklabels(np.arange(6)/5, fontsize=8)
    ax[1,0].legend(['System MTF', 'Diffraction-Limited MTF'],
                   fontsize=7)
    
    # Plot Ensquared and Encircled Energies.
        # Encircled
    ax[0,1].plot(1000*radii[radii<radMax], encirc[radii<radMax])
    ax[0,1].set_xlabel('Radius ($\mu$m)', fontsize=10)
    ax[0,1].set_ylabel('Energy', fontsize=10)
    ax[0,1].set_title('Encircled Energy')
    xticks = np.round(1000*np.linspace(0,radMax,6), 0)
    yticks = np.round(np.linspace(0,encirc[radii<radMax].max().item(),6), 2)
    ax[0,1].set_xticks(xticks)
    ax[0,1].set_yticks(yticks)
    ax[0,1].set_xticklabels(xticks, fontsize=8)
    ax[0,1].set_yticklabels(yticks, fontsize=8)
        # Ensquared
    ax[1,1].plot(1000*2*radii[2*radii<radMax], ensq[2*radii<radMax])
    ax[1,1].set_xlabel('Side of Square ($\mu$m)', fontsize=10)
    ax[1,1].set_ylabel('Energy', fontsize=10)
    ax[1,1].set_title('Ensquared Energy')
    xticks = np.round(1000*np.linspace(0,radMax,6), 0)
    yticks = np.round(np.linspace(0,ensq[2*radii<radMax].max().item(),6), 2)
    ax[1,1].set_xticks(xticks)
    ax[1,1].set_yticks(yticks)
    ax[1,1].set_xticklabels(xticks, fontsize=8)
    ax[1,1].set_yticklabels(yticks, fontsize=8)
    plt.show()
    
    # Plot PSF lineout.
    plotx = np.linspace(-crop_range,crop_range,psf_region.shape[0])
    plt.figure(dpi=300)
    plt.plot(plotx,psf_region[psf_region.shape[0]//2,:]/psf_region.max())
    plt.plot(plotx,psf_region[:,psf_region.shape[1]//2]/psf_region.max())
    plt.title('Lineout through PSF')
    plt.xlabel('Distance from center ($\mu$m)')
    plt.ylabel('PSF Intensity (Normalized to Peak)')
    plt.legend(['Horizontal Lineout', 'Vertical Lineout'])
    plt.show()
