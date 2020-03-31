# Vocal Similarity Scoring (VSS)

VSS is an automatic voice similarity scoring system.


## System requirements
At the time the VSS system was built we used the Keras (2.1.2) and tensorflow (1.2.0) libraries. We cannot guaranty that it would work out of the box with most current versions.


## How to
You first need to prepare the data using the make-trials.py script. The script creates numpy arrays for features vectors and trials given two vocal identities list files and their corresponding features (i-vector, x-vector, p-vector).

Simply run :
```
python train.py && test.py
```

Data can be shared on explicit request.
