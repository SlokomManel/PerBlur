## Towards User-Oriented Privacy for Recommender System Data: A Personalization-based Approach to Gender Obfuscation for User Profiles



This repository releases the Python implementation of our paper "*Towards User-Oriented Privacy for Recommender System Data: A Personalization-based Approach to Gender Obfuscation for User Profiles*": [paper](https://www.sciencedirect.com/science/article/pii/S0306457321002065).
PerBlur stands for Personalized Blurring. 
PerBlur is an amelioration of:
  * *BlurM(or)e* (published in RMSE@RecSys'2019 Workshop): [paper](https://pure.tudelft.nl/portal/files/68758824/short2.pdf), [code](https://github.com/STrucks/BlurMore),
  * *BlurMe* (published in RecSys'2012): [paper](https://ece.northeastern.edu/fac-ece/ioannidis/static/pdf/2012/blurme.pdf).
  
  
The figure below presents the framework that we use to carry out our analysis of gender obfuscation and demonstrate the properties of our PerBlur approach. As shown in Figure (top), gender obfuscation takes the original user-item matrix R and transforms it into the obfuscated user-item matrix R'.
In order to be successful, gender obfuscation must fulfill two criteria.
First, as indicated by Evaluation of Recommendation Performance in Figure (middle), the quality of the predictions produced by the recommender system must be comparable for the original and the obfuscated data.
Second, as indicated by Evaluation of the Extent to Which Gender Information is Blocked in Figure (bottom), a gender classifier must no longer be able to use the obfuscated data to reliably predict the genders of the users.


![Diagram](Diagram_PerBlur.png)

# Key Contributions :
* We introduce PerBlur, an approach that obfuscates recommender system data.
* PerBlur uses personalized blurring to block inference of users’ gender.
* We describe the user-oriented privacy paradigm in which PerBlur is formulated.
* We propose an evaluation procedure for obfuscated recommender system data.
* PerBlur is demonstrated to be capable of maintaining recommender system performance.
* We show the potential of obfuscation to improve fairness and diversity.


# Python packages to install:
* For fancyimpute package we need scikit_learn==0.21.0
* Numpy
* Pandas
* scikit-learn

For recommender system algorithms (BPRMF and ALS) we used lenskit toolkit [lkpy](https://github.com/lenskit/lkpy). Further documentation on how to use Lenskit package can be found [here](https://lkpy.readthedocs.io/en/stable/)

# Citation: 
```
@article{Slokom2021PerBlur,
title = {Towards user-oriented privacy for recommender system data: A personalization-based approach to gender obfuscation for user profiles},
author = {Manel Slokom and Alan Hanjalic and Martha Larson},
journal = {Information Processing & Management},
volume = {58},
number = {6},
pages = {102722},
year = {2021},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2021.102722},
url = {https://www.sciencedirect.com/science/article/pii/S0306457321002065},
}
```

# Contacts: 
Feel free to contact me at m.slokom@tudelft.nl or manel.slokom@live.fr for questions!
