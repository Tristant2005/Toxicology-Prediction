import pickle

import streamlit as st

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

# Chemistry Imports
from rdkit import Chem
from rdkit.Chem import Draw
import pubchempy as pcp
from rdkit.Chem import AllChem

# GLOBAL VARIABLES
CACHE_FILE = "feature_cache.pkl"
checkbox_keys = [ # Define checkbox labels and their session state keys
    "nr_ar_selected", "nr_er_selected", "sr_atad5",
    "nr_ar_lbd_selected", "nr_er_lbd_selected", "sr_hse_selected",
    "nr_ahr_selected", "nr_ppar_gamma_selected", "sr_mmp_selected",
    "nr_aromatase_selected", "sr_are_selected", "sr_p53_selected"
]

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Model Page", 
    page_icon="🤖",
    layout="wide"
)
def config():
    # Sidebar pages
    st.sidebar.page_link('pages/model.py', label='Model')
    st.sidebar.page_link('pages/about.py', label='About')

# LOCAL CACHE FUNCTIONS (Improves performance)""" 
def load_cache():
    #Load data from Pickle file
    try:
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)  # Returns data
    except FileNotFoundError:
        return {}  # Return empty dict if cache file doesn't exist
def save_cache(cache):
    #Save data using Pickle.
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

# CHEMISTRY FUNCTIONS (get_smile, get_features)""" 
def get_smile(name):
    try:
        c = pcp.get_compounds(name, 'name')
        smile = c[0].isomeric_smiles
    except:
        smile = ''
    
    return smile

@st.cache_data
def get_features_cached(smile):
    
    cache = load_cache()

    if (smile in cache):
        print(f"Loading {smile} from cache...")
        return cache[smile]  # return features found in cache
    
    print(f"Extracting features for {smile}...")
    features = get_features(smile)  # Compute features

    # Store in cache and save
    cache[smile] = features
    save_cache(cache)

    return features
def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bertModel(**inputs)

    # Extract the last hidden state of the first token (CLS token)
    feature_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return feature_vector

# MODEL FUNCTIONS
@st.cache_resource
def load_models_and_scaler():
    # Neural Nets
    nr_ar_neural = load_models('nr_ar.h5', False)
    nr_ar_lbd_neural = load_models('nr_ar_lbd.h5', False)
    nr_ahr_neural = load_models('nr_ahr.h5', False)
    nr_aromatase_neural = load_models('nr_aromatase.h5', False)
    nr_er_neural = load_models('nr_er.h5', False)
    nr_er_lbd_neural = load_models('nr_er_lbd.h5', False)
    nr_ppar_gamma_neural = load_models('nr_ppar_gamma.h5', False)
    sr_are_neural = load_models('sr_are.h5', False)
    sr_atad5_neural = load_models('sr_atad5.h5', False)
    sr_hse_neural = load_models('sr_hse.h5', False)
    sr_mmp_neural = load_models('sr_mmp.h5', False)
    sr_p53_neural = load_models('sr_p53.h5', False)

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
    scaler_nn = load_pkl("scalerNN.pkl")
    scaler_svm = load_pkl("scalerSVM.pkl")

    # LDA
    lda = load_pkl("lda.pkl")

    return (nr_ar_neural, nr_ar_lbd_neural, nr_ahr_neural, 
            nr_aromatase_neural, nr_er_neural, nr_er_lbd_neural, nr_ppar_gamma_neural,
            sr_are_neural, sr_atad5_neural, sr_hse_neural, sr_mmp_neural, sr_p53_neural,
            nr_ar_svm, nr_ar_lbd_svm, nr_ahr_svm, nr_aromatase_svm, nr_er_svm, nr_er_lbd_svm,
            nr_ppar_gamma_svm, sr_are_svm, sr_atad5_svm, sr_hse_svm, sr_mmp_svm, sr_p53_svm, scaler_nn, scaler_svm, lda)

def load_models(name, is_svm):
    # load model
    if (is_svm):
        path = 'models/SVMs/' + name
        model = joblib.load(path)
    else:
        path = 'models/NeuralNets/' + name # where it's stored on local computer
        model = load_model(path, compile=False)
    
    #print(model.summary())
    return model
def load_pkl(name):
    pkl_file = joblib.load('models/' + name)
    return pkl_file

# PREDICTION FUNCTIONS (Neural Nets and SVMs)
def predict_neural_nets(feature_data, model, scaler, lda):

    # Reshaping to 2D (1 sample, n features)
    feature_data_reshaped = feature_data.reshape(1, -1)

    # Prediction for Neural Nets
    normalized = scaler.transform(feature_data_reshaped)
    lda_normalized = lda.transform(normalized)

    # Get probability for the positive class (class 1)
    prediction = model.predict(lda_normalized)

    if (prediction >= 0.5): prediction = 1
    else: prediction = 0
    return prediction

def predict_svms(feature_data, model, scaler, lda):
    # Prediction code for SVMs
    normalized = scaler.transform(feature_data.reshape(1,-1))
    lda_normalized = lda.transform(normalized)
    prediction = model.predict(lda_normalized)

    if (prediction == 0): prediction = 0
    else: prediction = 1

    return prediction

def generate_prediction_explanation(model_name, prediction_value_svm, prediction_value_nn):
    """Generate interpretation based on model and prediction value"""
    if (model_name == "NR-AR"):
        if (prediction_value_svm == 1 and prediction_value_nn == 1):
            return "Prediction of 1 indicates the compound may interact with the androgen receptor, potentially affecting hormone signaling."
        elif (prediction_value_svm == 0 and prediction_value_nn == 0):
            return "Prediction of 0 suggests no significant interaction with the androgen receptor."
        else:
            return "Mixed predictions indicate uncertain androgen receptor interaction."
    elif (model_name == "NR-AR-LBD"):
        if (prediction_value_svm == 1 and prediction_value_nn == 1):
            return "Prediction of 1 indicates the compound may bind to the androgen receptor's ligand-binding domain, potentially affecting hormone signaling."
        elif (prediction_value_svm == 0 and prediction_value_nn == 0):
            return "Prediction of 0 suggests no significant binding to the androgen receptor's ligand-binding domain."
        else:
            return "Mixed predictions indicate uncertain androgen receptor binding activity."
    elif (model_name == "NR-AhR"):
        if (prediction_value_svm == 1 and prediction_value_nn == 1):
            return "Prediction of 1 suggests the compound may activate the aryl hydrocarbon receptor, which is involved in the body's response to environmental chemicals."
        elif (prediction_value_svm == 0 and prediction_value_nn == 0):
            return "Prediction of 0 suggests no significant activation of the aryl hydrocarbon receptor."
        else:
            return "Mixed predictions indicate uncertain aryl hydrocarbon receptor activation."
    elif (model_name == "NR-ER"):
        if (prediction_value_svm == 1 and prediction_value_nn == 1):
            return "Prediction of 1 suggests the compound may bind to the estrogen receptor, potentially affecting hormone regulation."
        elif (prediction_value_svm == 0 and prediction_value_nn == 0):
            return "Prediction of 0 suggests no significant interaction with the estrogen receptor."
        else:
            return "Mixed predictions indicate uncertain estrogen receptor interaction."
    elif (model_name == "NR-ER-LBD"):
        if (prediction_value_svm == 1 and prediction_value_nn == 1):
            return "Prediction of 1 suggests the compound may bind specifically to the estrogen receptor’s ligand-binding domain, possibly affecting hormone signaling."
        elif (prediction_value_svm == 0 and prediction_value_nn == 0):
            return "Prediction of 0 suggests no significant binding to the estrogen receptor’s ligand-binding domain."
        else:
            return "Mixed predictions indicate uncertain estrogen receptor binding activity."
    elif (model_name == "NR-PPAR-gamma"):
        if (prediction_value_svm == 1 and prediction_value_nn == 1):
            return "Prediction of 1 suggests the compound may activate PPAR-gamma, a receptor involved in metabolism and fat storage regulation."
        elif (prediction_value_svm == 0 and prediction_value_nn == 0):
            return "Prediction of 0 suggests no significant activation of PPAR-gamma."
        else:
            return "Mixed predictions indicate uncertain PPAR-gamma activation."
    elif (model_name == "SR-ARE"):
        if (prediction_value_svm == 1 and prediction_value_nn == 1):
            return "Prediction of 1 suggests the compound may activate the antioxidant response element (ARE), which is involved in oxidative stress response."
        elif (prediction_value_svm == 0 and prediction_value_nn == 0):
            return "Prediction of 0 suggests no significant activation of the antioxidant response element."
        else:
            return "Mixed predictions indicate uncertain ARE activation."
    elif (model_name == "SR-ATAD5"):
        if (prediction_value_svm == 1 and prediction_value_nn == 1):
            return "Prediction of 1 suggests the compound may interfere with ATAD5, a gene involved in DNA repair and genome stability."
        elif (prediction_value_svm == 0 and prediction_value_nn == 0):
            return "Prediction of 0 suggests no significant effect on ATAD5 activity."
        else:
            return "Mixed predictions indicate uncertain ATAD5 activity disruption."
    elif (model_name == "SR-HSE"):
        if (prediction_value_svm == 1 and prediction_value_nn == 1):
            return "Prediction of 1 suggests the compound may activate the heat shock response, which helps protect cells from stress-induced protein damage."
        elif (prediction_value_svm == 0 and prediction_value_nn == 0):
            return "Prediction of 0 suggests no significant activation of the heat shock response."
        else:
            return "Mixed predictions indicate uncertain heat shock response activation."
    elif (model_name == "SR-MMP"):
        if (prediction_value_svm == 1 and prediction_value_nn == 1):
            return "Prediction of 1 suggests the compound may disrupt mitochondrial membrane potential, which can affect energy production and apoptosis."
        elif (prediction_value_svm == 0 and prediction_value_nn == 0):
            return "Prediction of 0 suggests no significant effect on mitochondrial membrane potential."
        else:
            return "Mixed predictions indicate uncertain mitochondrial membrane disruption."
    elif (model_name == "SR-p53"):
        if (prediction_value_svm == 1 and prediction_value_nn == 1):
            return "Prediction of 1 suggests the compound may activate the p53 pathway, which is involved in DNA damage response and cell cycle regulation."
        elif (prediction_value_svm == 0 and prediction_value_nn == 0):
            return "Prediction of 0 suggests no significant activation of the p53 pathway."
        else:
            return "Mixed predictions indicate uncertain p53 pathway activation."
    else:
        return f"The prediction value of SVM-{prediction_value_svm} and NeuralNet-{prediction_value_nn} for {model_name} has no specific interpretation."


# UI FUNCTIONS
def toggle_checkboxes(): 
    # Function to toggle all checkboxes
    new_state = not st.session_state.get("all_selected", False)
    for key in checkbox_keys:
        st.session_state[key] = new_state
    st.session_state["all_selected"] = new_state

def main():    
    config()

    st.title("Model Page")

    # Grab all the models/scalers/pca that have been loaded
    (nr_ar_neural, nr_ar_lbd_neural, nr_ahr_neural, 
    nr_aromatase_neural, nr_er_neural, 
    nr_er_lbd_neural, nr_ppar_gamma_neural,
    sr_are_neural, sr_atad5_neural, sr_hse_neural, 
    sr_mmp_neural, sr_p53_neural,
    nr_ar_svm, nr_ar_lbd_svm, nr_ahr_svm, 
    nr_aromatase_svm, nr_er_svm, nr_er_lbd_svm,
    nr_ppar_gamma_svm, sr_are_svm, sr_atad5_svm, 
    sr_hse_svm, sr_mmp_svm, sr_p53_svm, scaler_nn, scaler_svm, lda) = load_models_and_scaler()

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
    molecule_embeddings = get_features_cached(smile)

    # Columns for checkboxes
    col1, col2, col3, col4 = st.columns(4)
    # Place checkboxes inside the columns
    with col1:
        st.checkbox("NR-AR Model", key="nr_ar_selected")
        st.checkbox("NR-ER Model", key="nr_er_selected")
        st.checkbox("SR-ATAD5 Model", key="sr_atad5")

    with col2:
        st.checkbox("NR-AR-LBD Model", key="nr_ar_lbd_selected")
        st.checkbox("NR-ER-LBD Model", key="nr_er_lbd_selected")
        st.checkbox("SR-HSE Model", key="sr_hse_selected")

    with col3:
        st.checkbox("NR-AhR Model", key="nr_ahr_selected")
        st.checkbox("NR-PPAR-gamma Model", key="nr_ppar_gamma_selected")
        st.checkbox("SR-MMP Model", key="sr_mmp_selected")

    with col4:
        st.checkbox("NR-Aromatase Model", key="nr_aromatase_selected")
        st.checkbox("SR-ARE Model", key="sr_are_selected")
        st.checkbox("SR-p53 Model", key="sr_p53_selected")

    # Button to toggle all checkboxes
    st.button("Select All / Deselect All", on_click=toggle_checkboxes)

    # Initialize an empty DataFrame for the predictions and interpretations
    prediction_df = pd.DataFrame(columns=["Task Name", "SVM Pred.", "NeuralNet Pred.", "Interpretation"])

    # Prediction Button
    if st.button("Predict"):
        if not any(st.session_state[key] for key in checkbox_keys):
            st.warning("Please select at least one model before predicting.")
        else:
            st.write("### Predictions:")

            with st.spinner('Running predictions...'):
                try:
                    # Dictionary of models and their corresponding predictions
                    models = {
                        "NR-AR": (st.session_state["nr_ar_selected"], nr_ar_svm, nr_ar_neural),
                        "NR-AR-LBD": (st.session_state["nr_ar_lbd_selected"], nr_ar_lbd_svm, nr_ar_lbd_neural),
                        "NR-AhR": (st.session_state["nr_ahr_selected"], nr_ahr_svm, nr_ahr_neural),
                        "NR-Aromatase": (st.session_state["nr_aromatase_selected"], nr_aromatase_svm, nr_aromatase_neural),
                        "NR-ER": (st.session_state["nr_er_selected"], nr_er_svm, nr_er_neural),
                        "NR-ER-LBD": (st.session_state["nr_er_lbd_selected"], nr_er_lbd_svm, nr_er_lbd_neural),
                        "NR-PPAR-gamma": (st.session_state["nr_ppar_gamma_selected"], nr_ppar_gamma_svm, nr_ppar_gamma_neural),
                        "SR-ARE": (st.session_state["sr_are_selected"], sr_are_svm, sr_are_neural),
                        "SR-ATAD5": (st.session_state["sr_atad5"], sr_atad5_svm, sr_atad5_neural),
                        "SR-HSE": (st.session_state["sr_hse_selected"], sr_hse_svm, sr_hse_neural),
                        "SR-MMP": (st.session_state["sr_mmp_selected"], sr_mmp_svm, sr_mmp_neural),
                        "SR-p53": (st.session_state["sr_p53_selected"], sr_p53_svm, sr_p53_neural),
                    }

                    # Loop through the models
                    for task_name, (is_selected, svm_model, nn_model) in models.items():
                        if is_selected:
                            svm_prediction = predict_svms(molecule_embeddings, svm_model, scaler_svm, lda)
                            neural_net_prediction = predict_neural_nets(molecule_embeddings, nn_model, scaler_nn, lda)
                            interpretation = generate_prediction_explanation(task_name, svm_prediction, neural_net_prediction)

                            print(neural_net_prediction)

                            # Create a DataFrame for the new row
                            new_row = pd.DataFrame({
                                "Task Name": [task_name],
                                "SVM Pred.": [svm_prediction],
                                "NeuralNet Pred.": [neural_net_prediction],
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