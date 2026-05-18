import streamlit as st

from rag_pipeline import retrieve_docs, generate_response

st.set_page_config(page_title="Upwork API Support Bot")

st.title("Upwork API Technical Support Bot")

query = st.text_input("Ask a question about the Upwork API")

if query:

    with st.spinner("Searching documentation..."):

        docs = retrieve_docs(query)

        answer, latency = generate_response(query, docs)

    st.subheader("AI Answer")

    st.write(answer)

    st.subheader("Latency")

    st.write(f"{latency} seconds")

    st.subheader("Sources")

    for i, doc in enumerate(docs, 1):
        st.markdown(f"### Source {i}")
        st.code(doc.page_content)