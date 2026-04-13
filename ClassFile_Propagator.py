# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 08:53:01 2026

@author: agilj
"""

import torch
from scipy.interpolate import LinearNDInterpolator

# Propagator Class
class propagator():
    """
    Parameters:
        device: str
            The device ('cpu' or 'cuda') that the model should run on.
    """
    def __init__(self, device):
        self.device = device
        
    """
    We superimpose the spherical waves propagating from the pupil, in the sense
    of the Huygen's Principal.
    Parameters:
        rayPos: torch.Tensor (N,3)
            The set of rays we are propagating.
        rayWave: torch.Tensor (N,3)
            The wave description (amp, phase) of each ray.
        Xobs: torch.Tensor (M,)
            The x-coordinates of the observation grid.
        Yobs: torch.Tensor (M,)
            The y-coordinates of the observation grid.
        wvl: float
            The wavelength of interest. (mm)
        z: float
            The distance we are propagating.
            
    Returns:
        torch.Tensor (M, M): The propagated wavefront.
    """
    def Huygens(self, rayPos, rayWave, obs_x, obs_y, wvl, z):
        k = 2 * torch.pi / wvl
        
        # Create observation plane meshgrid
        X_obs, Y_obs = torch.meshgrid(obs_x, obs_y, indexing='xy')
        X_obs = X_obs.T  # shape (Ny, Nx)
        Y_obs = Y_obs.T

        # Reshape rays for broadcasting
        xj = rayPos[:, 0].unsqueeze(-1).unsqueeze(-1)  # (N_rays,1,1)
        yj = rayPos[:, 1].unsqueeze(-1).unsqueeze(-1)
        Aj = rayWave[:, None, None]  # (N_rays,1,1)

        # Compute phase for all rays at all observation points
        R = torch.sqrt((X_obs[None,:,:] - xj)**2 + \
                       (Y_obs[None,:,:] - yj)**2 + \
                       z**2)
        phase = k * (R-z)  # (N_rays, Ny, Nx)

        # Sum contributions
        U = torch.sum(Aj * torch.exp(1j * phase), dim=0)  # (Ny, Nx)
    
        return U
    
    """
    ASM, but performed in a sparse sense to get better output control.
    
    Parameters:
        U: torch.Tensor (N,N)
            The complex wavefront.
        wvl: float
            The wavelength of interest. (mm)
        z: float
            The distance we are propagating.
        dxIn: float
            The input spatial sampling. (U spatial step size.)
        obx_x: torch.Tensor (M,)
            The output observation grid's x-coordinate.
        obs_y: torch.Tensor (M,)
            The output observation grid's y-coordinate
        chunkSize: int
            Only used if doChunk==True. The sidelength of the "chunk" whose
            inverse transform is computed at one time.
        doChunk: bool
            If True, computes the inverse transform in blocks, which are
            assembled to get the output wavefront.
    Returns:
        torch.Tensor (M,M): The propagated wavefront.
    """
    def ASM_sparse(self, U, wvl, z, dxIn, obs_x, obs_y, chunkSize=256,
                   doChunk=False):
        k = 2*torch.pi / wvl
        # Input spatial frequencies
        N = U.size()[0]
        fx = torch.fft.fftfreq(N, d=dxIn, device=self.device)
        fx_s = torch.fft.ifftshift(fx)
        FXin, FYin = torch.meshgrid(fx_s, fx_s, indexing='ij')
        dfx = 1 / (N*dxIn)
        # Fourier transform input
        Uft = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(U))) \
            * dxIn**2

        # Transfer function
        H = torch.exp(1j * k * z * torch.sqrt(1 - 
                                              (wvl*FXin)**2 - 
                                              (wvl*FYin)**2))
        H *= ((wvl*FXin)**2 + (wvl*FYin)**2 <= 1)  # evanescent waves
        # Fourier domain convolution
        Uft = Uft * H
        
        # Use if you have enough memory.
        if doChunk==False:
            # Inverse transform (sparse)
            Ey = torch.exp(1j * 2*torch.pi * obs_y[:,None] * fx_s[None,:])
            tmp = Ey @ Uft
            Ex = torch.exp(1j * 2*torch.pi * obs_x[:,None] * fx_s[None,:])
            U_out = tmp @ Ex.T
        
        # Use if you do not have enough memory.
        # Compute inverse transform in pieces so we can use more grid divs.
        if doChunk:
            U_out = torch.zeros((obs_x.size()[0], obs_y.size()[0]))
            for yind in range(0, obs_y.size()[0], chunkSize):
                Ey = torch.exp(1j * 2*torch.pi * \
                               obs_y[yind:yind+chunkSize,None] * fx_s[None,:])
                tmp = Ey @ Uft
                
                for xind in range(0, obs_x.size()[0], chunkSize):
                    Ex = torch.exp(1j * 2*torch.pi * \
                                   obs_x[xind:xind+chunkSize,None] * fx_s[None,:])
                    U_out[yind:yind+chunkSize,
                          xind:xind+chunkSize] = tmp @ Ex.T
        
        # Scaling factor
        U_out = U_out * (dfx**2)
        
        return U_out
    
    """
    Fraunhofer propagation. The PSF is the Fourier transform of the exit pupil
    after subtracting the reference sphere.
    
    Parameters:
        U: torch.Tensor (N,3)
            The set of rays we are propagating. They should be at the exit
            pupil for the Fraunhofer method.
        trans: torch.Tensor (N,3)
            The transmission coefficient that each ray has accumulated.
        W: torch.Tensor (N,3)
            The OPD for each ray (i.e. that ray's OPL minus the chief OPL).
        wvl: float
            The wavelength of interest. (mm)
        propDist: float
            The distance we are propagating.
        imgPoint: torch.Tensor (2,)
            The (x, y) position in the sampling plane the image is formed at,
            according to the raytrace.
        numObsPts: int
            The sidelength (in pixels) of the output sampling grid.
        obsSpacing: float
            The pixel pitch in the observation plane.
        gridDiv: int
            The sidelength (in pixels) of the exit pupil.
            
    Returns:
        torch.Tensor (numObsPts,numObsPts): The propagated wavefront.
    """
    def fraunhofer(self, rays, trans, W, wvl, propDist, imgPoint, numObsPts, 
                   obsSpacing, gridDiv=512):
        # Get some variables.
        k = 2 * torch.pi / wvl
        ex_x = rays[:, 0]
        ex_y = rays[:, 1]
        d_chief = torch.sqrt(imgPoint[0]**2 + imgPoint[1]**2 + propDist**2)

        # Get actual pupil size defined by rays.
        pupil_extent = torch.max(torch.abs(torch.cat([ex_x, ex_y])))
        dx_pupil = (2 * pupil_extent) / gridDiv

        x = torch.linspace(-pupil_extent, pupil_extent, gridDiv)
        X, Y = torch.meshgrid(x, x, indexing='xy')

        # Build triangulation once, interpolate twice
        points = torch.stack([ex_x, ex_y], dim=1).cpu().numpy()
        xi = (X.cpu().numpy(), Y.cpu().numpy())

        interp_W = LinearNDInterpolator(points, W.cpu().numpy(), fill_value=0)
        W_grid = torch.tensor(interp_W(*xi), dtype=torch.float32)
        A_grid = torch.tensor(
            LinearNDInterpolator(interp_W.tri, trans.cpu().numpy(), fill_value=0)(*xi),
            dtype=torch.float32
        )
                
        # Now build complex pupil function.
        pupil = A_grid * torch.exp(1j * k * W_grid.to(torch.float64))
        pupil = pupil.to(torch.complex128)
        pupil[torch.sqrt(X**2 + Y**2) > pupil_extent] = 0

        # Get number of points needed for FFT.
        N_total = int(wvl * d_chief / (obsSpacing * dx_pupil))
        N_total = max(N_total, gridDiv)
        N_total += (N_total % 2)

        # Pad pupil.
        pad = (N_total - gridDiv) // 2
        pupil_padded = torch.zeros((N_total, N_total), dtype=torch.complex128)
        pupil_padded[pad:pad+gridDiv, pad:pad+gridDiv] = pupil

        # Test 1: Your actual pupil -> what shift?
        pupil_real = A_grid * torch.exp(1j * k * W_grid)
        pupil_real[torch.sqrt(X**2 + Y**2) > pupil_extent] = 0
        
        # Test 2: Only tilt from your W_grid
        mask = A_grid > 0.01
        tilt_only = torch.zeros_like(W_grid)
        tilt_slope = 0.2822 * 0.55e-3  # waves/mm * wvl = mm/mm
        tilt_only = tilt_slope * Y
        pupil_tilt = torch.zeros_like(pupil_real)
        pupil_tilt[mask] = torch.exp(1j * k * tilt_only[mask]).to(torch.complex64)
        
        # Fraunhofer propagation.
        U_out = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(pupil_padded))
        )
        
        # Normalize wavefront and restrict to sampling region.
        center = int(N_total // 2)
        half = int(numObsPts // 2)
        U_out = U_out[center-half:center+half, center-half:center+half]
        U_out = U_out / U_out.sum()

        return U_out