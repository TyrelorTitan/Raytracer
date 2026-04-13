# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 08:51:31 2026

@author: agilj
"""

import numpy as np
import torch
import copy


# Ray Tracer Class
class raytracer():
    """
    Parameters:
        device: str
            The device ('cpu' or 'cuda') that the model runs on.
    """
    def __init__(self, device):
        self.device = device
               
    """
    Helper method that normalizes the input data.
    
    Parameters:
        v: torch.Tensor (...,N)
            The set of vectors to be normalized.
            
    Returns
        torch.Tensor (...,N): The normalized set of vectors.
    """
    def _normalize(self, v, eps=1e-12):
        nrm = torch.linalg.norm(v, dim=-1, keepdim=True)
        nrm = torch.clamp(nrm, min=eps)
        
        return v / nrm
    
    """
    Solve for intersection of ray r = r0 + t*u with sphere of radius R centered
    at center. There is a derivation for a general method to do this, but it
    boils down to a quadratic equation. We use the a, b, and c coefficients of
    a general quadratic to find the intersection points (if any).
    
    Parameters:
        r0: torch.Tensor (N,3)
            The starting point for each ray in the bundle.
        u: torch.Tensor (N,3)
            The direction unit vector for each ray.
        center: torch.Tensor (3,)
            The coordinates for the center of the sphere.
        R: float
            The radius of the sphere we want to intersect.
        
    Returns: 
        torch.Tensor (N,): Smallest positive t for which there is an 
                           intersection or torch.inf if no intersection.
    """
    def _raySphereIntersect(self, r0, u, center, R):
        # Get distance from each ray to the center of the sphere.
        oc = r0 - center
        # Calculate quadratic coefficients.
        a = torch.sum(u * u, dim=-1)
        b = 2.0 * torch.sum(u * oc, dim=-1)
        c = torch.sum(oc * oc, dim=-1) - (R * R)
        # Calculate discriminant.
        disc = b * b - 4.0 * a * c
        # Make array to hold the parameter values for intersection
        t = torch.full_like(disc, float('inf'), device=self.device)
        # Intersection only for non-negative discriminant.
        mask = disc >= 0.0
        if mask.any():
            # Compute intersection points.
            sqrt_d = torch.sqrt(disc[mask])
            a_m = a[mask]
            b_m = b[mask]
            t1 = (-b_m - sqrt_d) / (2.0 * a_m)
            t2 = (-b_m + sqrt_d) / (2.0 * a_m)

            # Make sure we don't run into numerical issues.
            t1[(t1<0) & (np.abs(t1)<1e-3)] = 0
            t2[(t2<0) & (np.abs(t2)<1e-3)] = 0
            
            # Pick smallest positive parameter.
            pos1 = t1 >= 0
            pos2 = t2 >= 0
            # This bit is complicated, but breaks down as:
            #   If two positive t values, pick the smaller.
            #   Else, if the first is positive, pick it.
            #   Else, if the second is positive, pick it.
            #   Else, put an infinity there.
            # Note that we need that last "else" to handle when the ray is
            # traveling *away* from the sphere, so the t's for intersection are
            # negative.
            tpos = torch.where(pos1 & pos2, 
                               torch.minimum(t1, t2), 
                               torch.where(pos1, 
                                           t1, 
                                           torch.where(pos2, 
                                                       t2, 
                                                       torch.full_like(t1, \
                                                               float('inf')))))
            if len(tpos) > 2:
                tpos[tpos > tpos.mean() + 5*torch.std(tpos)] = 0
            t[mask] = tpos
            
        return t
    
    """
    Solve for intersection of ray r = r0 + t*u with a plane normal to the
    optical axis.
    
    Parameters:
        r0: torch.Tensor (N,3)
            The starting point for each ray in the bundle.
        u: torch.Tensor (N,3)
            The direction unit vector for each ray.
        z_plane: float
            The z-location (along the optical axis) of the plane.
        
    Returns: 
        torch.Tensor (N,): Smallest positive t for which there is an 
                           intersection or torch.inf if no intersection.
    """
    def _rayPlaneIntersect(self, r0, u, z_plane):
        # Get z direction.
        uz = u[...,2]
        denom = uz.clone()
        # Avoid division by zero
        near_zero = torch.isclose(denom, torch.tensor(0.0, device=self.device))
        denom[near_zero] = 1e-20
        # Now we solve z_plane = z0 + t*u
        tz = (z_plane - r0[...,2]) / denom
        # Any negative points, set them to zero.
        tz = torch.where(tz < 0, torch.full_like(tz, 0), tz)
        
        return tz
    
    """
    Vectorized Snell refraction.
    
    Parameters:
        u: torch.Tensor (N,3)
            The direction unit vectors for each ray before refraction.
        n: torch.Tensor (N,3)
            The surface normal vector at the interface. By convention, n points
            *out of* the surface (so from medium 2 into medium 1).
        n1: float
            The index of refraction of the first medium.
        n2: float 
            The index of refraction of the second medium
    
    Returns:
        torch.Tensor (N,3): The direction unit vectors for transmitted rays.
                            In cases of TIR, returns nan.
        torch.Tensor (N,): Boolean mask that is True for for TIR.
    """
    def _refract(self, u, n, n1, n2):
        # Ensure we have unit vectors for ray direction and surface normals.
        u = self._normalize(u); 
        n = self._normalize(n)
        # Compute cosine of incidence angle.
        cos_i = - (u*n).sum(-1).clamp(-1,1)
        # Ratio of refractive indices.
        eta = torch.Tensor([n1/n2])
        # Snell's Law to get sine squared of transmission angle.
        sin_t2 = eta**2*(1-cos_i**2)
        # Where the sine is greater than 1 (due to refractive indices), we have
        # total internal reflection -- no light is transmitted.
        tir = sin_t2>1
        # Compute cosine of transmission angle (trig identity).
        cos_t = torch.sqrt((1-sin_t2).clamp(min=0))
        # Vector form of Snell's Law to get transmitted direction vector.
        t = eta.unsqueeze(-1)*u + (eta*cos_i - cos_t).unsqueeze(-1)*n
        # Normalize to get direction unit vector.
        t = self._normalize(t)
        # Set TIR rays to nan.
        t[tir] = torch.nan
        
        return t, tir 
    
    """
    Apply the aperture associated with a surface.
    
    Parameters:
        rays: torch.Tensor (N,3)
            The ray bundle passing through the system.
        ap: torch.Tensor (H, W)
            The aperture of the system.
        R: float
            The radius of curvature in mm of the surface with the aperture.
        diam: float
            The diameter of the aperture in mm.
        circCenter: float
            The location along the optical axis of the center of the sphere
            defining the surface.
            
    Returns:
        torch.Tensor (N,3): The transmission coefficients of each ray.
    """
    def _applyAperture(self, rays, ap, R, diam, circCenter):
        # Check that rays are not modulated by aperture.
        
        # CASES
        # -----
        # Aperture is not constant.
        # Aperture lies on curved surface.
        if (ap is not None) and \
            ((ap==ap[0,0]).all() == False) and \
                (np.isinf(R)==False):
            # Shift origin to center of sphere.
            rays_ap = copy.deepcopy(rays)
            rays_ap[:,2] = -R
            # Put rays into spherical coordinates.
            #   Rho -> R. theta, phi -> spatial coords.
            phi = torch.asin(rays_ap[:,0] / torch.norm(rays_ap,dim=1))
            theta = torch.asin(rays_ap[:,1] / torch.norm(rays_ap,dim=1))
            # Rays and aperture are now in same coord sys. Find which
            # pixel in the aperture each ray falls into.
                # Get the dAngle each pixel corresponds to.
            halfAngle = np.abs(np.asin(diam/(2*R)))
                # Build angle coordinate axis.
            axis = torch.linspace(0, 2*halfAngle, ap.shape[0]) - halfAngle
                # Find which pixel each ray intersects.
            phiInd = torch.searchsorted(axis, phi, side='left')
            thetaInd = torch.searchsorted(axis, theta, side='left')
                # Now multiply each ray by the corresponding axis value.
            trans = ap[thetaInd, phiInd] # switched for row/col convention.
            return trans
        # Aperture is not constant.
        # Aperture lies on flat surface.
        elif (ap is not None) and \
            ((ap==ap[0,0]).all() == False) and \
                (np.isinf(R)==True):
            # Rays and aperture start out in same coordinate system
            # (cartesian), so we can just find the intersects.
            x = rays[:,0]
            y = rays[:,1]
                # Build  coordinate axis.
            axis = torch.linspace(0, diam, ap.shape[0]) - diam/2
                # Find which pixel each ray intersects.
            axis = axis.contiguous()
            x = x.contiguous()
            y = y.contiguous()
            xInd = torch.searchsorted(axis, x, side='left')
            yInd = torch.searchsorted(axis, y, side='left')
                # Now multiply each ray by the corresponding axis value.
            trans = ap[xInd, yInd]
            return trans
        # Aperture is constant. Surface shape does not matter.
        elif (ap is not None) and ((ap==ap[0,0]).all() == True):
            trans = ap[0,0] # Constant aperture (ND-Filter)
            return trans
        
        return torch.ones(rays.shape[0])
    
    """
    Generate rays from a point source at infinity passing through a circular
    aperture.

    Parameters:
        pointSource: torch.Tensor (3,) 
            The position of the point source in mm.
        apMap: torch.Tensor (H W) 
            The front aperture of the system. Each element is a transmission
            coefficient on [0,1].
        apSize: float
            The width of the front aperture of the system.
            

    Returns:
        torch.Tensor (N,3): Tensor of ray origins (in pupil plane).
        torch.Tensor (N,3): Tensor of unit direction vectors.
        torch.Tensor (N,): Tensor of transmission weights.
    """
    def _sampleRaysAtAperture_inf(self, direction, apMap, apSize):
        H, W = apMap.shape
        apMap = apMap.to(self.device)
    
        # Create grid coordinates
        xs = torch.linspace(-apSize/2, 
                            apSize/2, 
                            W, 
                            device=self.device)
        ys = torch.linspace(-apSize/2, 
                            apSize/2, 
                            H, 
                            device=self.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        yy = yy.flatten()[:,None]
        xx = xx.flatten()[:,None]
        
        # Associate a transmission weight (from the aperture) with each ray.
        trans = apMap.flatten().to(float)
        
        # Remove any rays with weight 0 (no transmission)
        use = trans>0
        trans = trans[use]
        xx = xx[use]
        yy = yy[use]
        
        # One ray per pixel
        origins = torch.cat([xx, yy, torch.zeros(xx.shape[0], 
                                                 1, 
                                                 device=self.device)], 
                            dim=1)
        
        # Directions from source to pupil
        dirs = direction * torch.ones_like(origins)
    
        return origins, dirs, trans
    
    """
    Trace rays sequentially through the optical system.

    Parameters:
        rays: torch.Tensor (N,3)
            The starting positions for the rays.
        rayDirs: torch.Tensor (N,3)
            The ray direction unit vectors.
        trans: torch.Tensor (N,3)
            The transmission coefficient for each ray.
        train: list
            The optical system leading up to the sensor.
        wvl: float
            The wavelength of interest.
            
    Returns:
        dict: The final positions, directions, total OPL, valid ray indices, 
              and ray transmission percentages.
    """
    def _traceRays(self, rays, rayDirs, trans, train, wvl):
        opl = (rays[:,0]*rayDirs[:,0] + \
               rays[:,1]*rayDirs[:,1])
        
        valid = torch.ones(len(rays), 
                           dtype=torch.bool, 
                           device=self.device)
        n1 = 1 # Start in air.
        loc = 0
        
        for surfInd, surf in enumerate(train):
            # Get curvatures and thickness
            R = surf['curv']
            thick = surf['thic']
            diam = surf['diam']
            ap = surf['ap']
            circCenter = torch.Tensor([0, 
                                       0,
                                       loc + R]).to(self.device)
            n2 = surf['n'](wvl) # Going from n1 into the lens.

            # Find where each ray hits the surface.
            if R != np.inf:
                t = self._raySphereIntersect(rays, 
                                             rayDirs, 
                                             circCenter, 
                                             R)
            else:
                t = self._rayPlaneIntersect(rays, rayDirs, loc)
            valid = valid & (t!=torch.inf) # Only count good hits.
            pts = rays + rayDirs * t[:,None]
            # Check that points hit the physical lens.
            r2 = pts[:,0]**2 + pts[:,1]**2
            valid = valid & (r2 <= (diam/2)**2)
            # Apply aperture (if any).
            # Note that because we already checked for rays that missed the
            # surface, we treat the aperture as through it has diameter diam
            # which covers the horizontal width across the aperture.
            apTrans = self._applyAperture(rays[valid,:], ap, R, diam, circCenter)
            trans[valid] = trans[valid] * apTrans # Update total transmission coefficients.        
            # Get surface normals
            if R != np.inf:
                normals = (pts - circCenter) / R
                normals = self._normalize(normals)
            else:
                normals = torch.zeros_like(pts)
                normals[:,2] = -1
            # Count distance traveled toward's each ray's OPL.
            dL = torch.linalg.norm(pts - rays, dim=1)
            opl += n1 * dL
            # Now refract
            refrDirs, tir = self._refract(rayDirs,
                                          normals,
                                          n1,
                                          n2)
            
            ### Update all rays ###
            rays = pts
            rayDirs = refrDirs
            n1 = n2
            
            # Current z-location
            loc = loc + thick
            
        # Trace the rays to an imaginary plane (focal position).
        t = self._rayPlaneIntersect(rays, rayDirs, loc)
        valid = valid & (t!=torch.inf)
        pts = rays + rayDirs*t[:,None]
        dL = torch.linalg.norm(pts - rays, dim=1)
        opl += n1 * dL
        
        # Save the rays at the last pupil
        raysOut = pts

        # Don't use rays with 0 transmission.
        valid = valid & (trans!=0)

        return {
            'raysOut': raysOut,
            'rayDirsOut': rayDirs,
            'opl': opl,
            'valid': valid,
        }
    
    """
    Method to run the full ray-trace over the optical system.
    
    Parameters:
        train: list
            Each element, in order, in the optical train.
        inputDir: torch.Tensor (3,)
            The direction unit vector describing the rays' input angle.
        wvl: float
            The wavelength (in mm) of the input light.
            
    Returns:
        torch.Tensor (N,3): The rays at the exit pupil.
        torch.Tensor (N,3): The ray directions at the exit pupil.
        torch.Tensor (N,3): The OPL for each ray at the exit pupil.
        torch.Tensor (N,3): The transmission coefficients for each ray at the
                            exit pupil.
    """
    def runRayTrace(self, train, inputDir, wvl):         
        # Sample rays from the point source hitting the aperture.
        rays, rayDirs, trans = self._sampleRaysAtAperture_inf(inputDir,
                                                              train[0]['ap'],
                                                              train[0]['diam'])

        # Trace rays through optical system.
        traceOutputs = self._traceRays(rays, rayDirs, trans, train, wvl)
        rays = traceOutputs['raysOut']
        rayDirs = traceOutputs['rayDirsOut']
        opl = traceOutputs['opl']
        valid = traceOutputs['valid']

        # Filter out the invalid rays.
        if valid.sum() == 0:
            raise RuntimeError("No valid rays. Check lens system.")
        rays = rays[valid]
        rayDirs = rayDirs[valid]
        trans = trans[valid]
        opl = opl[valid]
        
        return rays, rayDirs, opl, trans