# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:50:17 2026

@author: agilj
"""

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import griddata


from ClassFile_GeomTracer import raytracer
from ClassFile_Propagator import propagator

class OpticalSys():
    """
    Parameters:
        curvList: list
            The list of each optical element's curvature.
        thickList: list
            The list of each optical element's center thickness.
        diamList: list
            The list of each optical element's diameter.
        apList: list
            The list of each optical element's aperture grid.
        device: str
            The device ('cpu' or 'cuda') that the model runs on.        
    
    Returns:
        None
    """
    def __init__(self, curvList, thickList, matList, diamList, apList,
                 device='cpu'):
        self.device = device
        
        n = self._getIndices(matList)
        
        # Define optical train.
        self.train = []
        for i in range(len(curvList)):
            self.train.append({'curv': curvList[i],
                               'thic': thickList[i],
                               'diam': diamList[i],
                               'ap': apList[i],
                               'n': n[i]})
            
        # Define ray tracer.
        self.raytracer = raytracer(self.device)
    
        # Define propagator
        self.prop = propagator(self.device)
    
    """
    Compute the PSF for the defined optical system. This is done via raytracing
    and wave-optics modeling. Relevant outputs related to the PSF are also
    output (i.e. the MTF and the PSF energy).
    
    Parameters:
        wvlList: list
            The wavelengths that should be considered when computing the PSF.
        inputDir: torch.Tensor (3,)
            The direction unit vector for the input rays.
        numObsPts: int
            The sidelength (in px) of the observation grid.
        obsSpacing: float
            The pixel pitch at the observation grid.
        defocus: float
            The amount of defocus (in mm) for which to compute the PSF.
        gridDiv: int
            The sidelength (in px) of the exit pupil.
        mode: str
            Either fraunhofer or sparseASM. The wave-optics method to use when
            computing the PSF. Note that fraunhofer is typically faster and
            performs well for smaller values of gridDiv (256 is a good start).
            sparseASM on the other hand, requires a far greater gridDiv (4096
            is a good start). sparseASM can also be more accurate for very high
            NA problems (such as in microscopy).
        
    Returns:
        list: The PSF for each wavelength.
        list: The MTF for each wavelength.
        torch.Tensor (N,): The x-axis for the MTFs.
        list: The ensquared energy for each PSF.
        list: The encircled energy for each PSF.
        torch.Tensor (M,): The x-axis for the energies.
        
    """
    def computePSF(self, wvlList, inputDir, numObsPts, obsSpacing, 
                   defocus=0, gridDiv=512, mode='fraunhofer'):
        # Estimate in-focus propagation distance
        wvl = wvlList[0]
        self._getABCD(wvl)
        self._deriveFromABCD()
        propDist = self.backFocalLength + defocus
        
        print('System EFL: '+str(self.efl.item()))
        print('System BFL: '+str(self.backFocalLength.item()))
        
        # Draw optical system.
        self.drawSystem(inputDir, wvl=wvlList[0], numRays=11)
        
        # Now make output lists.
        psfList = []
        mtfList = []
        ensqEnergyList = []
        encircEnergyList = []
        
        for wvl in wvlList:
            rays, rayDirs, opl, trans = self.raytracer.runRayTrace(self.train, inputDir, wvl)

            # After ray tracing, propagate rays to the image plane
            t_img = self.raytracer._rayPlaneIntersect(rays, rayDirs, rays[:,2].mean()+propDist)
            img_pts = rays + rayDirs * t_img[:, None]
            
            # Plot histogram
            histcenter = [torch.median(img_pts[:,0]*1000), torch.median(img_pts[:,1]*1000)]
            histrange = 10
            plt.figure(dpi=150)
            plt.hist2d(img_pts[:, 0].numpy()*1000, img_pts[:, 1].numpy()*1000, 
                       bins=200, range=[[histcenter[0]-histrange, histcenter[0]+histrange],
                                        [histcenter[1]-histrange, histcenter[1]+histrange]])
            plt.xlabel('X ($\mu$m)')
            plt.ylabel('Y ($\mu$m)')
            plt.title('Geometric PSF (Pure Ray Trace)')
            plt.axis('equal')
            plt.colorbar()
            plt.show()
            
            # Vertical lineout
            plt.figure(dpi=150)
            plt.hist(img_pts[:, 1].numpy()*1000, bins=300)
            plt.xlabel('Y ($\mu$m)')
            plt.title('Geometric PSF Vertical Lineout')
            plt.show()
            
            # Get the point where the image is formed by tracing chief ray.
            # Trace a single chief ray
            chief_origin = torch.zeros(1, 3)
            traceOutputs = self.raytracer._traceRays(chief_origin, 
                                                     inputDir[None,:], 
                                                     torch.Tensor([1]),
                                                     self.train, 
                                                     wvl)

            chief_pos = traceOutputs['raysOut']
            chief_dir = traceOutputs['rayDirsOut']
            chief_opl = traceOutputs['opl']
            chief_dir = chief_dir / chief_dir.norm()
            # imgPoint = chief_pos[0, :2]  # x, y at image plane
            imgPoint = chief_pos[0, :2] + propDist * (chief_dir[0, :2] / chief_dir[0, 2])
            
            # Plot OPD
            d_chief = torch.sqrt(imgPoint[0]**2 + imgPoint[1]**2 + propDist**2)

            # Subtract off reference sphere.
            ref_opl = torch.sqrt((rays[:,0] - imgPoint[0])**2 + 
                                 (rays[:,1] - imgPoint[1])**2 + 
                                 propDist**2)
            total_opl = opl + ref_opl
            chief_opl = chief_opl + d_chief
            W = total_opl - chief_opl

            print(f"On-axis OPD P-V: {(W-W.min()).max()/wvl:.2f} waves")

            # Plot OPD in waves.
            plt.scatter(rays[:,0].cpu(), rays[:,1].cpu(), c=(W-W.min()).cpu()/wvl, 
                        s=1)
            plt.gca().set_aspect('equal')
            plt.colorbar()
            plt.title('OPD in Waves')
            plt.xlabel('x-extent (mm)')
            plt.ylabel('y-extent (mm)')
            plt.show()
            
            print(f'Propagating {propDist.item():3f} mm.')
        
            if mode == 'fraunhofer':
                # This should probably be the default for 99% of problems.
                U_out = self.prop.fraunhofer(rays, trans, W, wvl, propDist, 
                                             imgPoint, numObsPts,
                                             obsSpacing, gridDiv=gridDiv)
                
            elif mode == 'sparseASM':
                # Define observation grid.
                # The ASM method shifts the PSF in x and y for reasons I 
                # can't quite figure out. I *think* it's inherent in the 
                # method, so as the number of grid divisions increases, it
                # converges to being not-shifted. This is a guess.
                obs_x = (torch.arange(numObsPts) - numObsPts/2) * \
                    obsSpacing
                obs_y = (torch.arange(numObsPts) - numObsPts/2) * \
                    obsSpacing
    
                # Sparse ASM does not use W, but the "raw" OPL.
                U, gridSize = self._binToExitPupil(rays, opl, wvl, 
                                                   gridDiv=gridDiv)
                dxIn = gridSize / gridDiv
                
                U_out = self.prop.ASM_sparse(U, wvl, propDist, dxIn, obs_x, 
                                             obs_y, chunkSize=256)
            # Compute PSF from complex field.
            psf = torch.abs(U_out)**2
            # Normalize PSF.
            psf = psf / psf.sum()
            # Get total ensquared and encircled energy.
            print('Computing Energies...')
            radii, ensq_E, encirc_E = self._getEnergy(psf, obsSpacing)
            # Compute MTF
            print('Computing MTF...')
            mtf, mtf_xAxis = self._getMTF(psf, obsSpacing)
            # Store things.
            psfList.append(psf)
            mtfList.append(mtf)
            ensqEnergyList.append(ensq_E)
            encircEnergyList.append(encirc_E)            
        
        return psfList, \
               mtfList, mtf_xAxis, \
               ensqEnergyList, encircEnergyList, radii
    
    """
    Bins the rays into a complex wavefront for use by sparse ASM propagation.
    
    Parameters:
        rays: torch.Tensor (N,3)
            The rays being traced.
        opl: torch.Tensor (N,3)
            The optical path length for each ray.
        wvl: float
            The wavelength being used.
        gridDiv: int
            The sidelength (in px) of the exit pupil.
            
    Returns:
        torch.Tensor (M,M): The output wavefront.
        float: The maximum width, in mm, of the output wavefront.
    """
    def _binToExitPupil(self, rays, opl, wvl, gridDiv=512):
        # Get grid size.
        minRay_x = rays[:,0].min()
        maxRay_x = rays[:,0].max()
        minRay_y = rays[:,1].min()
        maxRay_y = rays[:,1].max() 
        gridSize = torch.max(torch.Tensor((torch.abs(minRay_x), 
                                           torch.abs(maxRay_x), 
                                           torch.abs(minRay_y), 
                                           torch.abs(maxRay_y))))
        # Make grid.
        x = torch.linspace(0, 2*gridSize, gridDiv) - (gridSize)
        X, Y = torch.meshgrid(x, x, indexing = 'ij')
        # Bin OPLs on grid
        oplBin = torch.Tensor(griddata(rays[:,:2].cpu(),
                              opl.cpu(),
                              (X.cpu(),Y.cpu()),
                              method='linear', 
                              fill_value=0))
        k = 2*torch.pi/wvl
        U = torch.exp(1j*k*oplBin)
        
        U[torch.sqrt(X**2+Y**2)>gridSize] = 0
        
        return torch.Tensor(U), 2*gridSize

    """
    Computes the encircled and ensquared energies for the input PSF.
    
    Parameters:
        psf: torch.Tensor (N,N)
            The PSF under consideration.
        dx: float
            The pixel pitch in the observation plane.
    
    Returns:
        torch.Tensor (M,): The x-axis for the energies.
        torch.Tensor (M,): The encircled energy for each PSF.
        torch.Tensor (M,): The ensquared energy for each PSF.
    """
    def _getEnergy(self, psf, dx):
        N = psf.shape[0]
        total = psf.sum()
        center = N // 2
    
        x = (torch.arange(N) - center) * dx
        X, Y = torch.meshgrid(x, x, indexing='ij')
        R = torch.sqrt(X**2 + Y**2)
        S = torch.max(torch.abs(X), torch.abs(Y))
    
        num_pts = center
        radii = torch.linspace(0, R.max(), num_pts)
    
        # Encircled
        order_r = R.flatten().argsort()
        cum_r = (psf.flatten()[order_r] / total).cumsum(0)
        R_sorted = R.flatten()[order_r]
        encircled = torch.tensor(np.interp(radii.numpy(), R_sorted.numpy(), cum_r.numpy()))
    
        # Ensquared
        order_s = S.flatten().argsort()
        cum_s = (psf.flatten()[order_s] / total).cumsum(0)
        S_sorted = S.flatten()[order_s]
        ensquared = torch.tensor(np.interp(radii.numpy(), S_sorted.numpy(), cum_s.numpy()))
    
        return radii, encircled, ensquared
        
    """
    Uses ABCD transfer matrices to compute the back focal length of the lens
    system.
    
    Parameters:
        reference: bool
            A flag for whether to use the system's reference wavelength and 
            then store the result in the camera.
            
        wvl: Tensor (1,)
            The wavelength of interest.
            
    Returns:
        list : The A, B, C, and D coefficients of the system.
        
    """
    def _getABCD(self, wvl=0.55e-3):
        # Compute transfer matrices.
        # First lens outside the loop.
        n1 = 1
        n2 = self.train[0]['n'](wvl)
        R = self.train[0]['curv']
        thick = self.train[0]['thic']
        # Refraction at interface
        Mi_R = torch.Tensor([[         1,            0],
                              [(n1-n2)/(R*n2), n1/n2]]).to(self.device)
        # Travel through medium.
        Md_med = torch.Tensor([[1,  thick],
                               [0,    1     ]]).to(self.device)
        M = torch.mm(Md_med, Mi_R)
        # Now loop through rest of surfaces.
        for surfInd, surf in enumerate(self.train[1:], start=1):
            n1 = copy.deepcopy(n2)
            n2 = self.train[surfInd]['n'](wvl)
            R = self.train[surfInd]['curv']
            thick = self.train[surfInd]['thic']
            # Refraction at interface
            Mi_R = torch.Tensor([[         1,            0],
                                  [(n1-n2)/(R*n2), n1/n2]]).to(self.device)
            # Travel through medium.
            Md_med = torch.Tensor([[1,  thick],
                                   [0,    1     ]]).to(self.device)
            M = torch.mm(torch.mm(Md_med, Mi_R), M)
            
        # Now pull out A, B, C, D.
        A = M[0,0]
        B = M[0,1]
        C = M[1,0]
        D = M[1,1]
        
        self.ABCD = [A, B, C, D]
        
        return [A, B, C, D]
    
    """
    Uses computed ABCD matrices to calculate the physically meaningful
    quantities describing the optical system.
    
    Parameters:
        None
    
    Returns:
        dict: The EFL, FPP, RPP, FRL, and BFL, in mm, for this optical system.
    """
    def _deriveFromABCD(self):
        A, B, C, D = self.ABCD
        # Effective focal length
        self.efl = -1/C
        # Distance from first surface to front principal plane.
        self.frontPrincipalPlane = (1-A)/C
        # Distance from last surface to rear principal plane.
        self.rearPrincipalPlane = (D-1)/C
        # Focal distance from first surface to object.
        self.frontFocalLength = D/B
        # Focal distance from last surface to image.
        self.backFocalLength = -A/C
        
        return {'efl': self.efl,
                'frontPrincipalPlane': self.frontPrincipalPlane,
                'rearPrincipalPlane': self.rearPrincipalPlane,
                'frontFocalLength': self.frontFocalLength,
                'backFocalLength': self.backFocalLength}    
    
    """
    Get the indices we move into at each surface.
    
    Parameters:
        matList: list
            The list of materials the light passes into at each surface.
    
    Returns:
        list: A list of lambda functions corresponding to each index of refr.
              curve, such that if the wavelength is put into it, the index
              at that wavelength will be output.
    """
    def _getIndices(self, matList):
        n = []
        # Pull indices from file.
        for mat in matList:
            if mat == 'AIR':
                n.append(lambda x: 1)
            else:
                nTmp = self._getIndexFromFile('./indexFiles/'+mat+'.csv')
                n.append(nTmp)
        return n
                
    """
    Adds measured index of refraction data to the lens. This adds the 'n' entry 
    to self.lensParams. self.lensParams[lensInd]['n'] contains a lambda 
    function that returns the expected index of refraction for an input 
    wavelength.
    
    Parameters:
        glassFile: str
            The filepath to the .csv file containing a material's index of
            refraction curve.
    
    Returns:
        lambda: A lambda function corresponding to each index of refr. curve
                such that if the wavelength is put into it, the index at that 
                wavelength will be output.
    """
    def _getIndexFromFile(self, glassFile):
        if glassFile[-4:] == '.csv':
            df = pl.read_csv(glassFile,has_header=False)
            # Remove any missing data.
            df = df.drop_nulls()
            # Get all numeric data (as opposed to column headers) and create
            # a boolean column that indicates whether it is numeric or not.
            df = df.with_columns([
                pl.all_horizontal([
                    pl.col("column_1").cast(pl.Float64, strict=False)
                        .is_not_null(),
                    pl.col("column_2").cast(pl.Float64, strict=False)
                        .is_not_null()
                ]).alias("is_numeric")
            ])
            # Now group things between non-numeric entries.
            df = df.with_columns([
                    (~pl.col("is_numeric")).cast(pl.UInt32).cum_sum()\
                        .alias("group")
                 ])
            # Remove non-numeric entries.
            df_numeric = df.filter(pl.col("is_numeric")).select(["column_1", 
                                                                 "column_2", 
                                                                 "group"])
            # Convert grouped entries to arrays.
            arrays = [group.drop("group").cast(pl.Float64).to_numpy()
                      for _, group in df_numeric.group_by("group", 
                                                          maintain_order=True)
                     ]
            # The first array is the real part, the second is the imag part.
            nReal = torch.Tensor(arrays[0]).to(self.device)
            
            # Construct interpolation function for nReal.
            fInterp = lambda wvl: self._interp1d(nReal[:,0]*1e-3, #mm
                                                 nReal[:,1],
                                                 torch.Tensor([wvl])\
                                                     .to(self.device))
            return lambda wvl: fInterp(wvl)
        
    """
    Fast Pytorch linear interpolation function. Assumes the input xRef is 
    sorted.
    
    Parameters:
        xRef: torch.Tensor (N,)
            The independent variable of the reference function we are
            interpolating with respect to.
        yRef: torch.Tensor (N,)
            The dependent variable of the reference function we are
            interpolating with respect to.
        x: torch.Tensor (1,)
            The data point at which we want to know the interpolated y value.
            
    Returns:
        torch.Tensor (1,): The interpolated y-value corresponding to x.
    """
    def _interp1d(self, xRef, yRef, x):
        # Find where in reference the test point lies.
        ind = torch.searchsorted(xRef, x) - 1
        # ind = self.binary_search_gt(xRef, x) - 1
        if self.device == 'cuda:0':
            torch.cuda.synchronize()
        # If we are at the end, extrapolate.
        if ind == len(xRef)-1:
            ind -= 1
        # If we are at the beginning, extrapolate.
        if ind == -1:
            ind = 0
        # Get weight.
        t = (x - xRef[ind]) / (xRef[ind+1] - xRef[ind])
        # Interpolate.
        y = yRef[ind+1] + (1-t)*(yRef[ind]-yRef[ind+1])        
        return y
    
    """
    Method to compute the MTF of the scene given a centered PSF.
    
    Parameters:
        psf: torch.Tensor (N,N)
            The PSF for which we want to compute the MTF.
        dx: float
            The pixel pitch in the observation plane.
    Returns:
        torch.Tensor (N,N): The 2D MTF map for the input PSF.
        torch.Tensor (N,): The x-axis for the corresponding 1D MTF.
    """
    def _getMTF(self, psf, dx):
        # Get LSF
        lsf = psf.sum(dim=1)
        lsf_norm = lsf / lsf.sum()
        # Pad LSF for improved sampling rate.
        lsf_padded = F.pad(lsf_norm,[512,512])
        # Fourier Transform to get MTF.
        mtf = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(lsf_padded)))
        mtf = torch.abs(mtf).cpu()
        # Now get the x-axis for the MTF
        mtf_xAxis = torch.fft.fftshift(torch.fft.fftfreq(lsf_padded.size()[0], 
                                                          d=dx))
        return mtf, mtf_xAxis
    
    """
    Method to draw out the ray-trace, similar to OSLO and Zemax's system view.
    Note that I used Claude AI to assist in writing this method, since 
    drawing things out with matplotlib can be finicky sometimes..
    """
    def drawSystem(self, inputDir, wvl=0.55e-3, numRays=11):
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        
        # Need to make sure that we have already computed the ABCD matrices.
        if not hasattr(self.efl, 'item'):
            self._getABCD(wvl)
            self._deriveFromABCD()
        bfl = self.backFocalLength

        # Generate the rays we are going to use in the visualization.
        ap_diam = self.train[0]['diam']
        ray_y = torch.linspace(-ap_diam/2 * 0.95, ap_diam/2 * 0.95, numRays)
        ray_y = torch.tile(ray_y, (2,))
        rays = torch.zeros(2*numRays, 3)
        rays[:, 1] = ray_y
        rayDirs = torch.tile(inputDir, (numRays, 1))
        rayDirs = torch.cat((rayDirs, torch.zeros(numRays, 3)), dim=0)
        rayDirs[:, 2] = 1.0

        # Trace those rays, recording their positions at each surface.
        # Unfortunately, the recording requirement means that I have to
        # effectively replicate the ray tracing logic here.
        n1 = 1.0 # Start in air.
        loc = 0.0 # Define x=0 as the front aperture.
        all_positions = [rays.clone()]
        
        for surf in self.train:
            R = surf['curv']
            thick = surf['thic']
            n2 = surf['n'](wvl)
            circCenter = torch.Tensor([0, 0, loc + R])
            
            # Find intersects at next surface and propagate.
            if R != np.inf:
                t = self.raytracer._raySphereIntersect(rays, 
                                                       rayDirs, 
                                                       circCenter, 
                                                       R)
            else:
                t = self.raytracer._rayPlaneIntersect(rays, 
                                                      rayDirs, 
                                                      loc)
            pts = rays + rayDirs * t[:, None]
            
            # Get surface normals and refract rays.
            if R != np.inf:
                normals = self.raytracer._normalize((pts - circCenter) / R)
            else:
                normals = torch.zeros_like(pts)
                normals[:, 2] = -1
            refrDirs, tir = self.raytracer._refract(rayDirs, normals, n1, n2)
            
            # Save positions and update variables.
            all_positions.append(pts.clone())
            rays = pts
            rayDirs = refrDirs
            n1 = n2
            loc += thick
        
        # Propagate  to focal plane
        t = self.raytracer._rayPlaneIntersect(rays, 
                                              rayDirs, 
                                              loc + bfl)
        pts = rays + rayDirs * t[:, None]
        all_positions.append(pts.clone())
        
        # Get lens surface positions.
        z = 0
        z_positions = []
        for surf in self.train:
            z_positions.append(z)
            z += surf['thic']
        # Draw lens surfaces.
        for i, surf in enumerate(self.train):
            R = surf['curv']
            half_d = surf['diam'] / 2
            y = np.linspace(-half_d, half_d, 200)
            
            if R == np.inf:
                ax.plot([z_positions[i], z_positions[i]], [-half_d, half_d],
                        color='black', linewidth=1.2)
            else:
                sag = y**2 / (2 * R)
                ax.plot(z_positions[i] + sag, y, color='black', linewidth=1.2)
        
        # Color in glass regions.
        for i in range(len(self.train) - 1):
            n_val = self.train[i]['n'](wvl)
            if hasattr(n_val, 'item'):
                n_val = n_val.item()
            if n_val > 1.01:
                R1 = self.train[i]['curv']
                R2 = self.train[i + 1]['curv']
                diam = min(self.train[i]['diam'], self.train[i + 1]['diam'])
                half_d = diam / 2
                y = np.linspace(-half_d, half_d, 200)
                sag1 = y**2 / (2 * R1) if R1 != np.inf else np.zeros_like(y)
                sag2 = y**2 / (2 * R2) if R2 != np.inf else np.zeros_like(y)
                ax.fill_betweenx(y, z_positions[i] + sag1, z_positions[i+1] + sag2,
                                 color='#85B7EB', alpha=0.25, linewidth=0)
        
        # Draw apertures.
        max_d = max(s['diam'] for s in self.train) * 0.5
        for i, surf in enumerate(self.train):
            if surf['ap'] is not None:
                half_ap = surf['diam'] / 2
                ax.plot([z_positions[i], z_positions[i]], [half_ap, max_d],
                        color='black', linewidth=2.5)
                ax.plot([z_positions[i], z_positions[i]], [-half_ap, -max_d],
                        color='black', linewidth=2.5)
        
        # Draw rays. We use black for the chief rays and blue for the rays we
        # are interested in.
        colors = ['k']*numRays + ['b']*numRays
        positions = torch.stack(all_positions, dim=1)
        
        for ri in range(2*numRays):
            zs = positions[ri, :, 2].numpy()
            ys = positions[ri, :, 1].numpy()
            # Skip rays that went nan
            if np.any(np.isnan(ys)):
                last_valid = np.where(~np.isnan(ys))[0][-1]
                ax.plot(zs[:last_valid+1], ys[:last_valid+1], 
                        color=colors[ri], linewidth=0.7, alpha=0.7)
            else:
                ax.plot(zs, ys, color=colors[ri], linewidth=0.7, alpha=0.7)
        
        # Focal plane
        focal_z = z_positions[-1] + bfl
        ax.axvline(focal_z, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)
        ax.text(focal_z, -max_d - 0.5, 'focus', fontsize=7, ha='center', color='gray')
        
        # Format
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        margin = max_d * 1.3
        ax.set_ylim(-margin, margin)
        plt.tight_layout()
        plt.title('Paraxial Trace for Visualization')
        plt.show()