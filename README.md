# Raytracer
A ray tracer that I built to quickly test different optical systems with arbitrary apertures.

When the MainCall script is run, the PSF for a preset lens system is output, along with other relevant values.

Here is a brief example of the outputs, computed for the default lens system at 1-degree field angle.

The system's front aperture and paraxial trace:

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/d968dd53-3636-450a-a24d-f3779c545069" />
<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/807c6703-dcff-4f0a-9b2d-ee43b06f244a" />

A geometric ray trace:

<img width="500" height="375" alt="image" src="https://github.com/user-attachments/assets/2f349997-4119-4fcd-b964-dfeea48f5b0f" />

A geometric PSF lineout (vertical):

<img width="500" height="375" alt="image" src="https://github.com/user-attachments/assets/ed309473-6898-4e41-bb3d-0df1a22be88e" />

The exit pupil OPD:

<img width="500" height="480" alt="image" src="https://github.com/user-attachments/assets/f7d9e744-3a89-4801-b0ac-c261f32be84f" />

The predicted PSF on a 2 um pixel pitch sensor:

<img width="380" height="400" alt="image" src="https://github.com/user-attachments/assets/3a908734-e202-4d5b-86c7-a1ef9d0e2814" />

A summary output image:

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/e3e4dae6-1cb8-42d7-8a7e-fbb87164d684" />

Horizontal and vertical lineouts for the final PSF:

<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/b42aeb10-ae45-472b-910a-d63a91147f7e" />

As we can see in the above outputs, the geometric effects from a 1-degree shift are largely hidden, since diffraction effects expand the PSF to have a 20 micron width, but geometric effects are confined to only 2 or so microns.
If we increase the field angle to 4 degrees, we see that the PSF becomes highly aberrated as the geometric effects become more significant.

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/b1ff70c2-7d0e-45dc-b164-84f0632ecd99" />

To modify the optical system, open the MainCall script and scroll down to the parameters section.

<img width="701" height="684" alt="image" src="https://github.com/user-attachments/assets/4605e452-c146-4e74-bf3f-7c0bdbb59709" />

The raytracing parameters are given first and can be adjusted as needed to change the size of the observation grid, the observation grid pixel spacing, pupil divisions, field angle, etc.
The lens parameters section allows you to adjust the optical system and functions similarly to the surface tables in OSLO or Zemax. All distances and sizes are in mm unless otherwise specified.
To add new materials, a .csv file with the material index of refraction in it should be added to the indexFiles folder.
I got the ones currently in there by pulling the .csv from https://refractiveindex.info/, so the formatting is consistent with the outputs from that site.
The wvlList and QE lists allow the user to quickly loop through multiple wavelengths and corresponding sensor QE values to model specific sensors.
While the wvlList values are used in the raytracer, the QE values are not, and are only used when plotting the polychromatic PSF.
