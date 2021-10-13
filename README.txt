The .py files (e.g. TRAIN_GITHUB_16FEATURES.py) allow to train the networks and save reconstructed vibratory signals and latent representation.
They expect the vibratory signals are in the folder "./DATA/", in the working folder

Reconstructed signal and latent representations are saved as matlab arrays (i.e. ".mat" files) 

The number of features is specified in the file name (e.g. "_16FEATURES") and indicates the compression rate, i.e. features/256. This is because 256 is the length of the vibratory signals.

Output foders are created automatically. 

The reconstructed signals and the latent representations are saved as matlab files with the same name as the original signals but in different folders ("Reconstructed" and "Code", respectively)
