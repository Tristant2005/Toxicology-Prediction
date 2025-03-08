import streamlit as st

st.set_page_config(
    page_title="About", 
    page_icon="🤖",
    layout="wide"
)

# Page Layout
def config():
    st.sidebar.page_link('pages/model.py', label='Toxicology Prediction')
    st.sidebar.page_link('pages/about.py', label='About')

def main():

    config()

    st.title("About")
    st.write("Toxicology is a crucial field in modern medical science and an important factor in the synthesis of new materials, development of pharmaceuticals, and to evaulating the potency of pollutants.")
    st.write("However, toxicity is a difficult property to quantify, as it depends on the biochemistry and anatomy of the host organism. What is more important to chemists is not only discerning which molecules are toxic, but how they're toxic. For this reason, we define the toxicity of a molecule based off different biological responses of the body, each of which may point to a toxic effect.")

    st.divider() # Creates a horizontal line

    st.header("Tasks")
    st.write("For this project, the Tox21 Dataset is used. The tasks from the dataset is listed below, alongside the description of what each task represents.")

    st.markdown("""
    | Task Name | Description |
    |----------|----------|
    | NR-AR | Androgen Receptor: Protein that binds to androgens. |
    | NR-AR-LBD | Androgen Receptor LBD: Binding of androgenic compounds. |
    | NR-AhR | Aryl Hydrocarbon Receptor: Receptor in cell cytoplasm that detects Aryl Hydrocarbons. |
    | NR-Aromatase | Aromatase: Enzyme that converts androgens to estrogens. |
    | NR-ER  | Estrogen Receptor: Protein that binds to estrogen. |
    | NR-ER-LBD  | Estrogen Receptor LBD: Binding of estrogenic compounds. |
    | NR-PPAR-gamma  | Peroxisome Proliferator-Activated Receptor: Protein that regulates energy storage. |
    | SR-ARE  | Antioxidant Response Element: Defense against oxidative stress. |
    | SR-ATAD5  | ATAD5 gene: Involved in DNA damage response. |
    | SR-HSE  | Heat Shock Element: Triggers stress-response protein. |
    | SR-MMP  | Mitochondrial Membrane Potential: Used as an indicator for mitochondrial dysfunction. |            
    | SR-p53  | p53 Protein: Tumor suppressor involved in DNA repair and cell growth regulation. |
    """)

    st.divider() # Creates a horizontal line

    st.header("Featurization")

    st.write("To come up with a list of features for the molecules found in the Tox21 dataset, the CHEMBERT model was used. CHEMBERT is an opensource Chemical Language Transformer. It works by reading SMILES strings (molecular structure encoding) and converts these into useful patterns and features that describe the molecule’s properties. These features help us train models that can predict whether a molecule might be toxic.")


if __name__ == "__main__":
    main()