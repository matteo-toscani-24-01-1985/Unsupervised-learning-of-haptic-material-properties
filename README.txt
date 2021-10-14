The .py files (e.g. TRAIN_GITHUB_16FEATURES.py) allow to train the networks and save reconstructed vibratory signals and latent representation.
They expect the vibratory signals are in the folder "./DATA/", in the working folder

The folder "./DATA" needs to be extracted from the zipped DATA.zip file. 

each text file in the DATA folder corresponds to 10 seconds recordings of acceleration signals in g units. The file name indicates the category, the sample and the participant.  The category label is first, followed by '_', then 'SXX', where XX is the number of the matirial sample sample , 'CX' with X being the indext of the category corresponding to the initial category label (plastic= 1, paper = 2, fabric =3, animal=4,  stone=5, metal =6, wood=7), 'PX' with X being the participant number (1-11).

Each 10 seconds sample consists fo 32000 numbers.  For deep learning the singnals are segmented in 125 subsamples each of 256 numbers.

Reconstructed signal and latent representations are saved as matlab arrays (i.e. ".mat" files), with the same names of the original signals and a suffix indicating the subsample

There is one python file for each architecture. The number of features is specified in the file name (e.g. "_16FEATURES") and indicates the compression rate, i.e. features/256. This is because 256 is the length of the vibratory signals.

Output foders are created automatically. 

The reconstructed signals and the latent representations are saved as matlab files with the same name as the original signals but in different folders ("Reconstructed" and "Code", respectively)

The folder "IMAGES_LABELED" (compressed) contains photographs of each sample, with the same index as indicated in the filenames of the vibratory signals in "DATA".

Behavioral data are in the BEHAVIORAL.xlxs table. The first column indicates the category, the second the material sample (coded as in the filenames of the vibratory signals and the images), the third column indicates the participant number (1-11). The remaining column indicate the descriptors used for the ratings, with the same labels described in the manuscript. 

For any question do not exitate to contact us.

matteo_toscani@msn.com
metzger.anna@gmx.de
