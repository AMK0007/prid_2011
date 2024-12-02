Person Re-ID 2011 Dataset


General Information

The dataset consists of images extracted from multiple person trajectories 
recorded from two different, static surveillance cameras. Images from these 
cameras contain a viewpoint change and a stark difference in illumination, 
background and camera characteristics. Since images are extracted from 
trajectories, several different poses per person are available in each camera 
view. We have recorded 475 person trajectories from one view and 856 from the 
other one, with 245 persons appearing in both views. We have filtered out some 
heavily occluded persons, persons with less than five reliable images in each 
camera view, as well as corrupted images induced by tracking and annotation 
errors. This results in the following setup.
Camera view A shows 385 persons, camera view B shows 749 persons. The first 200 
persons appear in both camera views, i.e., person 0001 of view A corresponds to 
person 0001 of view B, person 0002 of view A corresponds to person 0002 of view 
B, and so on. The remaining persons in each camera view (i.e., person 0201 to 
0385 in view A and person 0201 to 0749 in view B) complete the gallery set of 
the corresponding view. Hence, a typical evaluation consists of searching the 
200 first persons of one camera view in all persons of the other view. This means 
that there are two possible evalutaion procedures, either the probe set is drawn 
from view A and the gallery set is drawn from view B (A to B, used in [1]), or 
vice versa (B to A).

Evaluation procedure A to B:
  - Probe set: the first 200 persons of A
  - Gallery set: all 749 persons of B

Evaluation procedure B to A:
  - Probe set: the first 200 persons of B
  - Gallery set: all 385 persons of A


Single-Shot / Multi-Shot

We provide two versions of the dataset, one representing the single-shot scenario 
and one representing the multi-shot scenario. The multi-shot version contains 
multiple images per person (at least five per camera view). The exact number 
depends on a person's walking path and speed as well as occlusions. The 
single-shot version contains just one (randomly selected) image per person 
trajectory, i.e., one image from view A and one image from view B.


Please cite the following paper if you use this dataset:

[1] Person Re-Identification by Descriptive and Discriminative Classification
Martin Hirzer, Csaba Beleznai, Peter M. Roth, and Horst Bischof
In Proc. Scandinavian Conference on Image Analysis, 2011
