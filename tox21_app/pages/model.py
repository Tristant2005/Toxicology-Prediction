import os
import pickle

import streamlit as st

import requests
import pandas as pd
import numpy as np

# Import/Load Model
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModel
import torch

# Load ChemBERTa tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
bertModel = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

from sklearn.preprocessing import StandardScaler
import joblib

# chemistry imports
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import Draw
import pubchempy as pcp
from rdkit.Chem import AllChem

CACHE_CHEMBERT_FILE = "feature_chembert_cache.pkl"
CACHE_MORDRED_FILE = "feature_mordred_cache.pkl"

st.set_page_config(
    page_title="Model Page", 
    page_icon="🤖",
    layout="wide"
)

# Page Layout
def config():
    st.sidebar.page_link('pages/model.py', label='Model')
    st.sidebar.page_link('pages/about.py', label='About')

# Local Cache Functions (Improves performance)
def load_chembert_cache():
    """Load data from Pickle file."""
    try:
        with open(CACHE_CHEMBERT_FILE, "rb") as f:
            return pickle.load(f)  # Returns data
    except FileNotFoundError:
        return {}  # Return empty dict if cache file doesn't exist
def save_chembert_cache(cache):
    """Save data using Pickle."""
    with open(CACHE_CHEMBERT_FILE, "wb") as f:
        pickle.dump(cache, f)

def load_mordred_cache():
    """Load data from Pickle file."""
    try:
        with open(CACHE_MORDRED_FILE, "rb") as f:
            return pickle.load(f)  # Returns data
    except FileNotFoundError:
        return {}  # Return empty dict if cache file doesn't exist
def save_mordred_cache(cache):
    """Save data using Pickle."""
    with open(CACHE_MORDRED_FILE, "wb") as f:
        pickle.dump(cache, f)

# Chemistry Functions (get_smile, get_features)
def get_smile(name):
    try:
        c = pcp.get_compounds(name, 'name')
        smile = c[0].isomeric_smiles
    except:
        smile = ''
    
    return smile
@st.cache_data
def get_features_mordred_cached(smile):

    cache = load_mordred_cache()

    if (smile in cache):
        print(f"Loading {smile} from cache...")
        return cache[smile]  # return features found in cache
    
    print(f"Extracting features for {smile}...")
    features = get_features_mordred(smile)  # Compute features

    # update caches with feature
    cache[smile] = features

    # Store in cache and save
    save_mordred_cache(cache)

    return features
def get_features_mordred(smile):
    '''
    This function transforms the smiles string into an rdkit mol object, then calculates relavent descriptors
    Descriptors are generated using mordred python library
    Mordred descriptors: https://mordred-descriptor.github.io/documentation/master/descriptors.html
    '''
    mol = AllChem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    #AllChem.EmbedMolecule(mol)
    AllChem.Compute2DCoords(mol)

    descriptor_list = {'nAcid', 'nBase', 'nAromAtom', 'nAtom', 'nSpiro', 'nBridgehead', 'nHetero', 'nB',
                        'nN', 'nO', 'nS', 'nP', 'nF', 'nCl', 'nBr', 'nI', 'nX', 'ATS0Z', 'ATS0p', 'AATS0dv',
                        'AATS0d', 'AATS0Z', 'AATS0v', 'AATS0i', 'ATSC1dv', 'ATSC2dv', 'ATSC3dv', 'ATSC4dv',
                        'ATSC5dv', 'ATSC6dv', 'ATSC7dv', 'ATSC8dv', 'ATSC2d', 'ATSC3d', 'ATSC4d', 'ATSC5d',
                        'ATSC6d', 'ATSC7d', 'ATSC8d', 'ATSC1Z', 'ATSC2Z', 'ATSC3Z', 'ATSC5Z', 'ATSC6Z',
                        'ATSC7Z', 'ATSC8Z', 'ATSC1v', 'ATSC2v', 'ATSC3v', 'ATSC4v', 'ATSC5v', 'ATSC6v',
                        'ATSC7v', 'ATSC8v', 'ATSC0p', 'ATSC2p', 'ATSC3p', 'ATSC4p', 'ATSC5p', 'ATSC6p',
                        'ATSC7p', 'ATSC1i', 'ATSC2i', 'ATSC3i', 'ATSC4i', 'ATSC5i', 'ATSC6i', 'ATSC7i',
                        'ATSC8i', 'AATSC0dv', 'AATSC0Z', 'AATSC0v', 'AATSC0i', 'BalabanJ', 'nBondsD',
                        'nBondsT', 'C2SP1', 'C1SP2', 'C3SP2', 'C1SP3', 'C3SP3', 'C4SP3', 'FCSP3',
                        'Xch-3d', 'Xch-5d', 'Xc-4d', 'Xc-5d', 'Xc-4dv', 'Xc-6dv', 'NssssB', 'NsCH3',
                        'NdCH2', 'NtCH', 'NdsCH', 'NsssCH', 'NddC', 'NaaaC', 'NssssC', 'NsNH3', 'NsNH2',
                        'NdNH', 'NssNH', 'NaaNH', 'NsssNH', 'NdsN', 'NaaN', 'NsssN', 'NddsN', 'NaasN',
                        'NssssN', 'NsOH', 'NssO', 'NaaO', 'NsssSiH', 'NssssSi', 'NsssP', 'NsssssP', 'NsSH',
                        'NdS', 'NssS', 'NaaS', 'NdssS', 'NddssS', 'NssssGe', 'NsssAs', 'NsssdAs', 'NdSe',
                        'NssSe', 'NaaSe', 'NdssSe', 'NssssSn', 'SsssB', 'SsssCH', 'SdssC', 'SaasC', 'fMF',
                        'nHBDon', 'IC0', 'IC1', 'MIC0', 'Lipinski', 'GhoseFilter', 'FilterItLogS', 'PEOE_VSA2',
                        'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
                        'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'SMR_VSA6', 'SMR_VSA9', 'SlogP_VSA1',
                        'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA7', 'SlogP_VSA10', 'EState_VSA2', 'EState_VSA3',
                        'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
                        'VSA_EState1', 'VSA_EState7', 'VSA_EState9', 'n4Ring', 'n5Ring', 'n7Ring', 'n8Ring', 'n9Ring',
                        'n10Ring', 'n11Ring', 'n12Ring', 'nG12Ring', 'nHRing', 'n6HRing', 'n8HRing', 'n12HRing', 'n3aRing',
                        'n4aRing', 'n5aRing', 'n7aRing', 'nG12aRing', 'naHRing', 'n6aHRing', 'nARing', 'n5ARing', 'n5AHRing',
                        'nFRing', 'n6FRing', 'n7FRing', 'n8FRing', 'n9FRing', 'n10FRing', 'n11FRing', 'n12FRing', 'nG12FRing',
                        'n7FHRing', 'n10FHRing', 'nG12FHRing', 'n9FaRing', 'n12FaRing', 'nG12FaRing', 'nG12FaHRing', 'nFARing',
                        'n9FARing', 'n10FARing', 'nRot', 'JGI1', 'JGI2', 'JGI3', 'JGI4', 'JGI5', 'JGI6', 'JGI7', 'JGI8', 'JGI9', 'JGI10'}

    calc = Calculator(descriptors, ignore_3D=True)  # register all descriptors
    calc.descriptors = [d for d in calc.descriptors if str(d) in descriptor_list]
    all_desc = calc.pandas([mol])

    return all_desc

@st.cache_data
def get_features_chembert_cached(smile):
    
    cache = load_chembert_cache()

    if (smile in cache):
        print(f"Loading {smile} from cache...")
        return cache[smile]  # return features found in cache
    
    print(f"Extracting features for {smile}...")
    features = get_features_chembert(smile)  # Compute features

    # Store in cache and save
    cache[smile] = features
    save_chembert_cache(cache)

    return features
def get_features_chembert(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bertModel(**inputs)

    # Extract the last hidden state of the first token (CLS token)
    feature_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return feature_vector

# Model Functions
@st.cache_resource
def load_models_and_scaler():
    # Neural Nets
    nr_ar_neural = load_models('bestNR_ARmodel.h5', False)

    # SVMs
    nr_ar_svm = load_models('NR_AR.pkl', True)
    nr_ar_lbd_svm = load_models('NR_AR_LBD.pkl', True)
    nr_ahr_svm = load_models('NR_AhR.pkl', True)
    nr_aromatase_svm = load_models('NR_Aromatase.pkl', True)
    nr_er_svm = load_models('NR_ER.pkl', True)
    nr_er_lbd_svm = load_models('NR_ER_LBD.pkl', True)
    nr_ppar_gamma_svm = load_models('NR_PPAR_gamma.pkl', True)
    sr_are_svm = load_models('SR_ARE.pkl', True)
    sr_atad5_svm = load_models('SR_ATAD5.pkl', True)
    sr_hse_svm = load_models('SR_HSE.pkl', True)
    sr_mmp_svm = load_models('SR_MMP.pkl', True)
    sr_p53_svm = load_models('SR_p53.pkl', True)

    # Scalers
    scaler = load_pkl("scaler.pkl")
    scaler_svm = load_pkl("scalerSVM.pkl")

    # PCA
    pca = load_pkl("PCA.pkl")

    return (nr_ar_neural, nr_ar_svm, nr_ar_lbd_svm, nr_ahr_svm, nr_aromatase_svm, nr_er_svm, nr_er_lbd_svm,
            nr_ppar_gamma_svm, sr_are_svm, sr_atad5_svm, sr_hse_svm, sr_mmp_svm, sr_p53_svm, scaler, scaler_svm, pca)

def load_models(name, is_svm):
    # load model
    if (is_svm):
        path = 'models/SVMs/' + name
        model = joblib.load(path)
    else:
        path = 'models/NeuralNets/' + name # where it's stored on local computer
        model = load_model(path)
    
    # model.summary()
    return model
def load_pkl(name):
    scaler = joblib.load('models/' + name)
    return scaler

# Prediction Functions (Neural Nets and SVMs)
def predict_neural_nets(feature_data, model, scaler):
    # Normalize Data
    normalized = scaler.transform(feature_data)

    prediction = model.predict(normalized)
    return prediction

def predict_svms(feature_data, model, scaler, pca):
    # Prediction code for SVMs
    normalized = scaler.transform(feature_data.reshape(1,-1))
    pca_normalized = pca.transform(normalized)
    pca_normalized = pca_normalized[:,:266]
    prediction = model.predict(pca_normalized)
    return prediction

def generate_prediction_explanation(model_name, prediction_value):
    """Generate interpretation based on model and prediction value"""
    if (model_name == "NR-AR"):
        if (prediction_value == 1):
            return "NR-AR prediction of 1 indicates toxicity for androgen receptor activity."
        else:
            return "NR-AR prediction of 0 indicates no significant androgren receptor toxicity."
    elif (model_name == "NR-AR-LBD"):
        if (prediction_value == 1):
            return "NR-AR-LBD prediction of 1 indicates toxicity for androgen receptor ligand binding domain activity."
        else:
            return "NR-AR-LBD prediction of 0 indicates no significant androgen receptor ligand binding domain toxicity."
    elif (model_name == "NR-AhR"):
        if (prediction_value == 1):
            return "NR-AhR prediction of 1 indicates toxicity for Aryl Hydrocarbon Receptor activity."
        else:
            return "NR-AhR prediction of 0 indicates no significant Aryl Hydrocarbon Receptor toxicity."
    elif (model_name == "NR-Aromatase"):
        if (prediction_value == 1):
            return "NR-Aromatase prediction of 1 indicates toxicity for Aromatase activity."
        else:
            return "NR-Aromatase prediction of 0 indicates no significant Aromatase toxicity."
    elif (model_name == "NR-ER"):
        if (prediction_value == 1):
            return "NR-ER prediction of 1 indicates toxicity for Estrogen Receptor activity."
        else:
            return "NR-ER prediction of 0 indicates no significant Estrogen Receptor toxicity."
    elif (model_name == "NR-ER-LBD"):
        if (prediction_value == 1):
            return "NR-ER-LBD prediction of 1 indicates toxicity for Estrogen Receptor Ligand Binding Domain activity."
        else:
            return "NR-ER-LBD prediction of 0 indicates no significant Estrogen Receptor Ligand Binding Domain toxicity."
    elif (model_name == "NR-PPAR-gamma"):
        if (prediction_value == 1):
            return "NR-PPAR-gamma prediction of 1 indicates toxicity for Peroxisome Proliferator-Activated Receptor Gamm activity."
        else:
            return "NR-PPAR-gamma prediction of 0 indicates no significant Peroxisome Proliferator-Activated Receptor Gamm toxicity."
    elif (model_name == "SR-ARE"):
        if (prediction_value == 1):
            return "SR-ARE prediction of 1 indicates toxicity for Antioxidant Response Element activity."
        else:
            return "SR-ARE prediction of 0 indicates no significant Antioxidant Response Element."
    elif (model_name == "SR-ATAD5"):
        if (prediction_value == 1):
            return "SR-ATAD5 prediction of 1 indicates toxicity for ATAD5 gene activity."
        else:
            return "SR-ATAD5 prediction of 0 indicates no significant ATAD5 gene toxicity."
    elif (model_name == "SR-HSE"):
        if (prediction_value == 1):
            return "SR-HSE prediction of 1 indicates toxicity for Heat Shock Element."
        else:
            return "SR-HSE prediction of 0 indicates no significant Heat Shock Element toxicity."
    elif (model_name == "SR-MMP"):
        if (prediction_value == 1):
            return "SR-MMP prediction of 1 indicates toxicity for Mitochondrial Membrane Potential activity."
        else:
            return "SR-MMP prediction of 0 indicates no significant Mitochondrial Membrane Potential toxicity."
    elif (model_name == "SR-p53"):
        if (prediction_value == 1):
            return "SR-p53 prediction of 1 indicates toxicity for p53 activity."
        else:
            return "SR-p53 prediction of 0 indicates no significant p53 toxicity."
    else:
        return f"The prediction value of {prediction_value} for {model_name} has no specific interpretation."

def main():    
    config()

    st.title("Model Page")

    # Grab all the models/scalers/pca that have been loaded
    (nr_ar_neural, nr_ar_svm, nr_ar_lbd_svm, nr_ahr_svm, 
     nr_aromatase_svm, nr_er_svm, nr_er_lbd_svm,
    nr_ppar_gamma_svm, sr_are_svm, sr_atad5_svm, 
    sr_hse_svm, sr_mmp_svm, sr_p53_svm, scaler, scaler_svm, pca) = load_models_and_scaler()

    # Type in a compound name
    name = st.text_input("Enter a compund name")

    if (not name):
        st.warning("Please enter a compound name.")
        return

    # Get SMILES string
    smile = get_smile(name)

    if (not smile):
        st.error(f"'{name}' is not listed in the database.")
        return # Return early - Don't show the following unless a smile is typed in
    else:
        st.write("Your SMILES string is:", smile)

    # Convert SMILES to molecule and display image
    chemical = Chem.MolFromSmiles(smile)
    if (chemical):
        img = Draw.MolToImage(chemical)
        st.image(img, caption="Molecule Image")

    # Get descriptors for molecule
    molecule_mordred = get_features_mordred_cached(smile)
    molecule_chembert = get_features_chembert_cached(smile)

    # Columns for checkboxes
    col1, col2, col3, col4 = st.columns(4)
    # Place checkboxes inside the columns
    with (col1):
        nr_ar_selected = st.checkbox("NR-AR Model")
        nr_er_selected = st.checkbox("NR-ER Model")
        sr_atad5 = st.checkbox("SR-ATAD5 Model")
    with (col2):
        nr_ar_lbd_selected = st.checkbox("NR-AR-LBD Model")
        nr_er_lbd_selected = st.checkbox("NR-ER-LBD Model")
        sr_hse_selected = st.checkbox("SR-HSE Model")
    with (col3):
        nr_ahr_selected = st.checkbox("NR-AhR Model")
        nr_ppar_gamma_selected = st.checkbox("NR-PPAR-gamma Model")
        sr_mmp_selected = st.checkbox("SR-MMP Model")
    with (col4):
        nr_aromatase_selected = st.checkbox("NR-Aromatase Model")
        sr_are_selected = st.checkbox("SR-ARE Model")
        sr_p53_selected = st.checkbox("SR-p53 Model")

    # Initialize an empty DataFrame for the predictions and interpretations
    prediction_df = pd.DataFrame(columns=["Task Name", "Prediction Value", "Interpretation"])

    # Prediction Button
    if st.button("Predict"):
        if (not nr_ar_selected and not nr_ar_lbd_selected and not nr_ahr_selected 
            and not nr_aromatase_selected and not nr_er_selected and not nr_er_lbd_selected
            and not nr_ppar_gamma_selected and not sr_are_selected and not sr_atad5
            and not sr_hse_selected and not sr_mmp_selected and not sr_p53_selected):
            st.warning("Please select at least one model before predicting.")
        else:
            st.write("### Predictions:")

            with st.spinner('Running predictions...'):
                try:
                    # Dictionary of models and their corresponding predictions
                    models = {
                        "NR-AR": (nr_ar_selected, nr_ar_svm),
                        "NR-AR-LBD": (nr_ar_lbd_selected, nr_ar_lbd_svm),
                        "NR-AhR": (nr_ahr_selected, nr_ahr_svm),
                        "NR-Aromatase": (nr_aromatase_selected, nr_aromatase_svm),
                        "NR-ER": (nr_er_selected, nr_er_svm),
                        "NR-ER-LBD": (nr_er_lbd_selected, nr_er_lbd_svm),
                        "NR-PPAR-gamma": (nr_ppar_gamma_selected, nr_ppar_gamma_svm),
                        "SR-ARE": (sr_are_selected, sr_are_svm),
                        "SR-ATAD5": (sr_atad5, sr_atad5_svm),
                        "SR-HSE": (sr_hse_selected, sr_hse_svm),
                        "SR-MMP": (sr_mmp_selected, sr_mmp_svm),
                        "SR-p53": (sr_p53_selected, sr_p53_svm),
                    }

                    # Loop through the models
                    for task_name, (is_selected, model) in models.items():
                        if is_selected:
                            prediction_value = predict_svms(molecule_chembert, model, scaler_svm, pca)
                            interpretation = generate_prediction_explanation(task_name, prediction_value)

                            # Create a DataFrame for the new row
                            new_row = pd.DataFrame({
                                "Task Name": [task_name],
                                "Prediction Value": [prediction_value],
                                "Interpretation": [interpretation]
                            })

                            # Concatenate the new row to the existing DataFrame
                            prediction_df = pd.concat([prediction_df, new_row], ignore_index=True)

                    # Convert the DataFrame to HTML with text wrapping for the Interpretation column
                    html_table = prediction_df.to_html(escape=False, index=False)
                    
                    # Add custom CSS for text wrapping and centering
                    st.markdown(f"""
                        <style>
                        .dataframe {{
                            width: 100%;
                            border-collapse: collapse;
                        }}
                        .dataframe th, .dataframe td {{
                            text-align: center;
                            padding: 10px;
                        }}
                        .dataframe td {{
                            white-space: normal !important;
                            word-wrap: break-word;
                        }}
                        </style>
                        """, unsafe_allow_html=True)

                    # Display the table with wrapped text
                    st.markdown(html_table, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"An error occured: {str(e)}")

if __name__ == "__main__":
    main()