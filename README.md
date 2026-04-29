# **B**outiqu**E** **L**ight curves for **A**steroseismology (BELA) with TESS


For smaller TESS candidate samples, it is recommended that you use a boutique method to construct light curves for each target rather than adopting pipeline generated products. Specifically, the aperture masks for each star should be individualised, as there is a higher potential of photometric contamination from neighbouring stars with large pixel scale of the TESS instrument. 

The BELA-TESS python package allows provides an interactive widget to test different alternative aperture masks, and simultaneously compare the resultant light curves. This pipeline also detrends the light curves that is optimised for asteroseismology, following the method in Howell et al. 2026. A python notebook tutorial is provided to step through the usage of this package. A video showing an example of the intereactive widget is shown here:

https://github.com/user-attachments/assets/653d0432-793d-4d3a-8d26-1727132ce6ad

BELA-TESS also provides an interactive widget for tests of the stitching of the sector light curves. Some sectors might show anomolous noise that could impact the oscillation frequencies in the power spectrum. This widget allows for you to test if removing certain sectors improves the final power spectrum. Again, example usage is provided in the python notebook tutorial, and a video example shown below:

https://github.com/user-attachments/assets/23f2f14c-40cf-40cb-91ea-a914703deb1d


This code is still under-developement. The following is a list of proposed additions. If you have any suggestions, please contact Maddy Howell (Howell.753@osu.edu)
TODO:
* Citation to paper
* Add in SNR comparison in the FFI widget
* Add another widget to compare power spectrum when masking out part of light curve
