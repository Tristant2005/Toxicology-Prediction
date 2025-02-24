import streamlit as st

st.set_page_config(
    page_title="About Page", 
    page_icon="🤖",
    layout="wide"
)

# Page Layout
def config():
    st.sidebar.page_link('pages/model.py', label='Model')
    st.sidebar.page_link('pages/about.py', label='About')

def main():

    config()

    st.title("About Page")
    st.write("Give a brief description of the project. Mention all the different models that we created, etc.")

    st.divider() # Creates a horizontal line

    st.header("Tasks")
    st.write("Talk about the tasks, a high level description of what the tasks represent related to chemistry.")

    st.markdown("""
    | Task Name | Description |
    |----------|----------|
    | NR-AR | Androgen Receptor |
    | NR-AR-LBD | Androgen Receptor Ligand Binding Domain |
    | NR-AhR | Aryl Hydrocarbon Receptor (response to environmental toxins) |
    | NR-Aromatase | Aromatase (enzyme that converts androgens to estrogens) |
    | NR-ER  | Estrogen Receptor |
    | NR-ER-LBD  | Estrogen Receptor Ligand Binding Domain |
    | NR-PPAR-gamma  | Peroxisome Proliferator-Activated Receptor Gamm (fat storage and glucose) |
    | SR-ARE  | Antioxidant Response Element |
    | SR-ATAD5  | The ATAD5 gene is involved in DNA damage response (could lead to mutations) |
    | SR-HSE  | Heat Shock Element (initiate heat shock responses) |
    | SR-MMP  | Mitochondrial Membrane Potential (indicator of mitochondrial dysfunction) |            
    | SR-p53  | p53 is a crucial tumor suppressor |
    """)

    st.divider() # Creates a horizontal line

    st.header("Models")
    st.write("Click on any model below for more information.")

    with st.expander("Model 1"):
        st.write("Explain model 1")

    with st.expander("Model 2"):
        st.write("Explain model 2")

    with st.expander("Model 3"):
        st.write("Explain model 3")

    with st.expander("Model 4"):
        st.write("Explain model 4")

if __name__ == "__main__":
    main()