# Toxicology-Prediction
A model that uses the tox21 dataset [1] to attempt to find a realtionship between molecular structure and realtive toxicity, and a streamlit website to use to interact with it.

This is currently still an ongoing project for the University of Victoria's AI club.

What still needs to be completed:
  - Make minimum viable product (MVP) by connecting first model to the streamlit site
  - Get a model that can train on three tasks
      - Fix problem with class weights

The dataset has already been modified to include molecular descriptors using mordred, however futher testing with other featurisation methods may be beneficial.

[1] https://tox21.gov/resources/

