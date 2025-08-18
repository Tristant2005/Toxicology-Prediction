# **Toxicology-Prediction: ⚛️**
A model that uses the tox21 dataset [1] to attempt to find a realtionship between molecular structure and realtive toxicity, and a streamlit website to use to interact with it.
This was a project made in the University of Victoria's AI club and was presented at the Canadaian Conference of Artificial Intelligence (CUCAI) 2025.

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

Tasks are the 12 classes each chemical may fall into, and a molecule can be multiple classifications.

# Inspiration
A study from the American Society for Biochemistry and Molecular Biology reports that 90% of drugs fail clinical testing, with 30% of failures attributed to molecular toxicity [2]. These failures result in billions of dollars in losses for pharmaceutical companies due to the costs of research, development, and testing. Our project focuses on this subset by developing a machine learning model capable of predicting molecular toxicity based solely on molecular structure and similarity to known toxic compounds, using the chemistry principle of Structure-Activity Relationship (SAR) as reasoning.

# Repo Structure
  - Refer to the _Notebooks_ folder to view model creation and analysis.
  - Refer to the _tox21dataset_ to replicate/perform any machine learning anaylsis of your own.
  - Refer to the _tox21app_ to view the implementation between a user interface and the many models.

# App Interface
The image below provides a look into our user interface for the many models.

![appDemo](https://github.com/user-attachments/assets/fb010207-6dfd-4921-b6f6-c2ba7d0520b6)

# Key Colaborators
* Tristan Tucker: https://www.linkedin.com/in/tristan-tucker-4751722a0/
* Simran Cheema: https://www.linkedin.com/in/simran-cheema-690755231/

# References:
  [1] National Center for Advancing Translational Sciences, "Toxicology in the 21st Century (Tox21)," 
  Tox21, [Online]. Available: https://tox21.gov/. [Accessed: Feb. 20, 2025].
  
  [2] D. Sun, "90% of drugs fail clinical trials," ASBMB Today, Mar. 12, 2022. [Online]. Available: 
  https://www.asbmb.org/asbmb-today/opinions/031222/90-of-drugs-fail-clinical-trials. [Accessed: Feb. 20, 2025].

