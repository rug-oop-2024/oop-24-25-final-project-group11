import streamlit as st
from app.core.system import AutoMLSystem


# Initialize AutoMLSystem singleton
automl = AutoMLSystem.get_instance()

# Initialize page state in session
if 'page' not in st.session_state:
    st.session_state['page'] = 'main'


def go_to_page(page_name) -> None:
    """
    Function to navigate to a specific page.
    """
    st.session_state['page'] = page_name


# Page title
st.title("Dataset Management")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.button("Manage Datasets", on_click=lambda: go_to_page('main'))
st.sidebar.button("Upload & Create Dataset", on_click=lambda: go_to_page('create'))
st.sidebar.button("Save Dataset", on_click=lambda: go_to_page('save'))

# Display different pages based on the current page state
if st.session_state['page'] == 'main':
    st.header("Available Datasets")

    # Fetch and display datasets from the registry
    datasets = automl.registry.list(type="dataset")

    if datasets:
        for dataset in datasets:
            st.write(f"- {dataset.name}")
    else:
        st.write("No datasets available.")

elif st.session_state['page'] == 'create':
    # Display the create page
    import app.datasets.management.create as create_page
    create_page.main()

elif st.session_state['page'] == 'save':
    # Display the save page
    import app.datasets.management.save as save_page
    save_page.main()
