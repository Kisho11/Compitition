import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Set up the model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text-generation pipeline with adjusted parameters
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, truncation=True,
                pad_token_id=tokenizer.eos_token_id)

# Set up LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()


def process_document(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    temp_file_path = f"temp{file_extension}"

    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())

    if file_extension == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension in [".docx", ".doc"]:
        loader = Docx2txtLoader(temp_file_path)
    else:
        os.remove(temp_file_path)
        return "Unsupported file format. Please upload a PDF or Word document."

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("vectors")

    os.remove(temp_file_path)
    return f"{file_extension[1:].upper()} processed and embeddings saved successfully!"


def get_qa_chain():
    loaded_vectors = FAISS.load_local("vectors", embeddings, allow_dangerous_deserialization=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=loaded_vectors.as_retriever(search_kwargs={"k": 2}),
        max_tokens_limit=900
    )
    return qa_chain


def generate_answer(question, context):
    prompt = f"Question: {question}\n\nContext: {context}\n\nProvide a concise and relevant answer based only on the given context. If the information is not available in the context, state that you don't have enough information to answer the question.\n\nAnswer:"
    response = pipe(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
    answer = response.split("Answer:")[-1].strip()
    return answer


def main():
    st.title("Enhanced Banking Assistant")

    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or Word document", type=["pdf", "docx", "doc"])
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                result = process_document(uploaded_file)
            st.success(result)

    st.header("Ask Questions")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                qa_chain = get_qa_chain()
                context = qa_chain.retriever.get_relevant_documents(prompt)
                context_text = " ".join([doc.page_content for doc in context])
                full_response = generate_answer(prompt, context_text)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                full_response = "I apologize, but I encountered an error while processing your question. Please try again or rephrase your question."

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
