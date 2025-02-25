# **Toxicology-Prediction: ⚛️**
A model that uses the tox21 dataset [1] to attempt to find a realtionship between molecular structure and realtive toxicity, and a streamlit website to use to interact with it.
This is currently still an ongoing project for the University of Victoria's AI club.

# About:
The Tox21 dataset includes a wide range of molecules and 12 "tasks" representing a quality of cellular biology and biochemistry

NR-XXX -> Nuclear Receptor  
SR-XXX -> Stress Response

```python
'''
['NR-AR',           Androgen Receptor  
 'NR-AR-LBD',       Androgen Receptor Ligand Binding Domain  
 'NR-AhR',          Aryl Hydrocarbon Receptor (response to environmental toxins)  
 'NR-Aromatase',    Aromatase (enzyme that converts androgens to estrogens)  
 'NR-ER',           Estrogen Receptor  
 'NR-ER-LBD',       Estrogen Receptor Ligand Binding Domain  
 'NR-PPAR-gamma',   Peroxisome Proliferator-Activated Receptor Gamm (fat storage and glucose)  

 'SR-ARE',          Antioxidant Response Element  
 'SR-ATAD5',        The ATAD5 gene is involved in DNA damage response (could lead to mutations)  
 'SR-HSE',          Heat Shock Element (initiate heat shock responses)  
 'SR-MMP',          Mitochondrial Membrane Potential (indicator of mitochondrial dysfunction) 
 'SR-p53']          p53 is a crucial tumor suppressor
'''
```

Tasks are the 12 axis each chemical may fall into, and a molecule can be multiple classifications.

# What still needs to be completed:
The following includes a few items currently worked on:
  - Improve streamlit site with implementation of the multi_classification model
  - Improve model to train on seven tasks instead of three
  - Fix issue relating to overfitting

# Repo Structure
  - Refer to the _Notebooks_ folder to view model creation and analysis.
  - Refer to the _tox21dataset_ to replicate/perform any machine learning anaylsis of your own.
  - Refer to the _tox21app_ to view the implementation between a user interface and the many models.

# References:
  [1] https://tox21.gov/resources/

