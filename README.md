# Toxicology-Prediction
A model that uses the tox21 dataset [1] to attempt to find a realtionship between molecular structure and realtive toxicity, and a streamlit website to use to interact with it.

This is currently still an ongoing project for the University of Victoria's AI club.

What still needs to be completed:
  - Normalize the dataset using SKlearn's standard scalar
  - Change the way the model trains by altering validation splits and sample wieghts
  - Inclusion of the Drugbank api to convert user-inputed molecular names into SMILE format (Streamlit site)

The dataset has already been modified to include molecular descriptors using mordred, however futher testing with other featurisation methods may be beneficial.

[1] https://tox21.gov/resources/

