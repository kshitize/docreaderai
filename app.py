 
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os


def main():
    st.header('Chat with PDF')
    load_dotenv()
    pdf_path = "./test.pdf"

    pdf_reader = PdfReader(open(pdf_path, "rb"))
    num_pages = len(pdf_reader.pages) 

    # Read the PDF content and store it in a variable
    pdf_content = ""
    for page in range(num_pages):
        pdf_content += pdf_reader.pages[page].extract_text()
    
    if pdf_content:
        st.text(pdf_content)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text=pdf_content)
        st.write(chunks)

        
        # embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
        # VectorStore = FAISS.from_texts(chunks,embedding=embeddings)

        store_name="test"
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
            st.write("Embeddings loaded from the disk")
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)
            st.write("Embedding Compution completed")

        query = st.text_input("Ask Question about PDF File")
        # st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query,k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            st.write(response)



if __name__=='__main__':
    main()