# **B**outiqu**E** **L**ight curves for **A**steroseismology (BELA) with TESS


For smaller TESS candidate samples, it is recommended that you use a boutique method to construct light curves for each target rather than adopting pipeline generated products. Specifically, the aperture masks for each star should be individualised, as there is a higher potential of photometric contamination from neighbouring stars with large pixel scale of the TESS instrument. 

The BELA-TESS python package allows provides an interactive widget to test different alternative aperture masks, and simultaneously compare the resultant light curves. This pipeline also detrends the light curves that is optimised for asteroseismology, following the method in Howell et al. 2026. A video showcasing an example of the intereactive widget is shown here:

https://github.com/user-attachments/assets/dd0197d0-0b16-41c6-8678-c19fa4a8fe2b


BELA-TESS also provides an interactive widget for tests of the stitching of the sector light curves. Some sectors might show anomolous noise that could impact the oscillation frequencies in the power spectrum. This widget allows for you to test if removing certain sectors improves the final power spectrum. Again, example usage is provided in the python notebook tutorial, and a video example shown below:

https://github.com/user-attachments/assets/9dde4fb3-e554-4191-8ffd-0dddcd25a9f0

## Usage
BELA-TESS requires the following inputs to create aperture masks for each sector:
* TIC ID
* Gaia ID
* Gmag
* (RA, Dec)
* (pmra, pmdec)
* estimate for $\nu_{\rm max}$
* TESS cutout (target pixel file) centred on target star

You can query MAST for TESS cutouts using the lightkurve python package. 

To stitch light curves together, BELA-TESS requires an array of light curve objects, the TIC ID and estimate for $\nu_{\rm max}$

## Example Jupyter Notebook
An example tutorial is provided for red giant star TIC 461599427 (same star as in the provided videos). This tutorial provides a working example of the required inputs for each module in BELA-TESS, that can be copied and applied to different stars observered by TESS.

## Required python packages:
* numpy
* scipy
* astropy
* lightkurve
* matplotlib
* pandas
* astroquery

## Citing
If you use `BELA-TESS`, please include the following citation [Howell et al., 2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260427828H/abstract)
```tex
@ARTICLE{Howell-2026,
       author = {{Howell}, Madeline and {Johnson}, Jennifer A. and {Pinsonneault}, Marc H. and {Morales}, Leslie M. and {Tayar}, Jamie and {Roberts}, John D. and {Stello}, Dennis and {McKenzie}, Madeleine},
        title = "{TESS Asteroseismology of Red Giants in the Old Metal-Rich Open Clusters NGC 188 \& NGC 6791}",
      journal = {arXiv e-prints},
     keywords = {Solar and Stellar Astrophysics, Astrophysics of Galaxies},
         year = 2026,
        month = apr,
          eid = {arXiv:2604.27828},
        pages = {arXiv:2604.27828},
          doi = {10.48550/arXiv.2604.27828},
archivePrefix = {arXiv},
       eprint = {2604.27828},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260427828H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Development
This code is still actively under-development. If you have any suggestions, please contact Maddy Howell (Howell.753@osu.edu)
