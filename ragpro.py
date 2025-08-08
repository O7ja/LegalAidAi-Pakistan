#!/usr/bin/env python
# coding: utf-8

# In[26]:


print("hi")

# In[58]:


import faiss
import streamlit

import warnings
warnings.filterwarnings("ignore")

# In[10]:


from langchain_community.document_loaders import PyMuPDFLoader
pdf_path ="C:/Users/DELL/Desktop/Rag/data/bla.pdf"
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()


# In[ ]:




# In[11]:


doc = documents[0]
print(doc.page_content)

# In[12]:



import os
pdfs= []
for root, dirs, files in os.walk("C:/Users/DELL/Desktop/Rag/data"):
    print(root , dirs, files,)
    for file in files:
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(root, file))


from langchain_community.document_loaders import PyMuPDFLoader

all_docs = []
for pdf_path in pdfs:
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    all_docs.extend(docs)  # Add pages from this PDF to the master list
            

# In[13]:


for i, doc in enumerate(all_docs[:4]):  # just show first 4 to limit output
    print(f"\n-----------Document {i+1}---------------")
    print(f"Metadata: {doc.metadata}")
    print(f"Page Content: {doc.page_content[:200]}...\n")  # Show first 200 chars


# In[14]:


docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    pages = loader.load()
    docs.extend(pages)

# In[15]:


len(docs)

# In[16]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)


# In[17]:


len(chunks) , len(docs)

# In[18]:


from langchain_ollama import OllamaEmbeddings
import faiss 
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


# In[19]:


embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434") 
single_vector = embeddings.embed_query("this is some text data")

# In[20]:


print(len(single_vector))
single_vector

# In[21]:


index = faiss.IndexFlatL2(len(single_vector))
index.ntotal , index.d

# In[22]:


len(chunks)

# In[24]:


vector_store = FAISS(
    embedding_function= embeddings,
    index = index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
    
)



# In[25]:


ids = vector_store.add_documents(documents=chunks)


# In[26]:


len(chunks)


# In[27]:


vector_store.index_to_docstore_id
len(ids)

  

# In[39]:


question = "What is the main topic of the document?"
docs = vector_store.search(query=question , search_type="similarity")
for doc in docs:
    print(doc.page_content)
    print("\n\n")

# In[40]:


retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k":100, "lambda_mult": 1,})


# In[41]:


docs = retriever.invoke(question)

# In[42]:


question = "What is pakistans law on freedom of speech?"
docs = retriever.invoke(question)

# In[36]:


from langchain import hub
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough


# In[ ]:




# In[63]:


from langchain_community.chat_models import ChatOllama
model = ChatOllama(model="gemma:2b", base_url="http://localhost:11434")


# In[64]:


prompt = hub.pull("rlm/rag-prompt")

# In[65]:


print (prompt)


# In[70]:


prompt = """
You are an assistant for a question-answering task.  
Use the provided context chunks to answer the question.  
If the answer is not present in the context, respond with "I don't know" in a friendly manner.  
Please format your answer as bullet points.  
Only use the given context to generate your answer — stay relevant.

Question: {question}

Context:
{context}

Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt)

# In[71]:


def format_docs(docs):
    return"\n\n".join({doc.page_content for doc in docs})

print(format_docs(docs))

# In[73]:



rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# In[ ]:


question = "What are fundamental rights guaranteed by the Constitution of Pakistan?"
output = rag_chain.invoke(question)
print(output)

# In[60]:


import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- UI Title ---
st.title("📄 Pakistan Law Chatbot")

# --- User Input ---
question = st.text_input("Ask a question about the legal PDFs:")

# --- When question is entered ---
if question:
    with st.spinner("Thinking..."):
        # Run the RAG chain
        answer = rag_chain.invoke(question)
        st.success(answer)


# In[57]:



