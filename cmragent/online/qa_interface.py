import streamlit as st
from agents.agent import CMR
from tools import load_llm_tools
from llm.utils import get_llm

def qa_interface(model_name, temperature, api_key, secret_key=None, system=None, token=None, appid=None,
                 api_secret=None, spark_url=None):
    st.title("Q&A and Chat Interface")

    if 'agent' not in st.session_state:
        try:
            llm = get_llm(
                model_name=model_name,
                temperature=temperature,
                api_key=api_key,
                secret_key=secret_key,
                system=system,
                token=token,
                appid=appid,
                api_secret=api_secret,
                spark_url=spark_url
            )
            tools = load_llm_tools(llm, verbose=True)
            st.session_state.agent = CMR(llm=llm, tools=tools)
            st.success(f"Model '{model_name}' initialized successfully.")
        except Exception as e:
            st.error(f"Error initializing model: {e}")
            return

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello, I am your assistant. How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        if "thought_process" in msg and msg["thought_process"].strip():
            with st.expander("Execution Process", expanded=False):
                st.code(msg["thought_process"], language=None)

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        try:
            response = st.session_state.agent.run(prompt)

            if isinstance(response, dict):
                answer = response.get("answer", "No answer found")
                thought_process = response.get("thought_process", "")
            else:
                answer = response
                thought_process = ""

            if thought_process.strip():
                message = {
                    "role": "assistant",
                    "content": answer,
                    "thought_process": thought_process
                }
            else:
                message = {
                    "role": "assistant",
                    "content": answer
                }

            st.session_state.messages.append(message)
            st.chat_message("assistant").write(answer)

            if thought_process.strip():
                with st.expander("Execution Process", expanded=True):
                    st.code(thought_process, language=None)

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")