import streamlit as st
from src.LLMs.groqllm import GroqLLM
from src.graph.graph_builder import GraphBuilder
from src.ui.streamlitui.loadui import LoadStreamlitUI
from src.ui.streamlitui.display_result import DisplayResultStreamlit

def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph Agentic AI application with the Streamlit UI.

    This function:
    - Initializes the UI
    - Handles user input
    - Configures the LLM
    - Sets up the graph based on the selected use case
    - Displays the output
    - Implements exception handling for robustness
    """
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()
    
    if not user_input:
        st.error('Error: Failed to load the user input from the UI')
        return
    # usercase = user_input.get('selected_usecase')
    # if not usercase:
    #     st.error('Error: No use case selected')
    #     return
    
    user_message = st.chat_input("Enter your message:")
    
    if user_message:
        try:
            obj_llm_config = GroqLLM(user_control_input=user_input)
            model  = obj_llm_config.get_llm_model()
            if not model:
                st.error('Error: Failed to load the model')
                return
            usecase = user_input.get('selected_usecase')
            if not usecase:
                st.error('Error: No use case selected')
                return
            graph_builder = GraphBuilder(model)
            try:
                graph = graph_builder.setup_graph(usecase)
                DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()
                
            except Exception as e:
                raise ValueError(f"Error occured with exception: {e}")    
        except Exception as e:
            raise ValueError(f"Error occured with exception: {e}")
    