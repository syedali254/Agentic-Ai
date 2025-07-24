from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from langchain.prompts import PromptTemplate

# Set up API key and LLM
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)  # use available model name

# Load and process the text file
loader = TextLoader("file1.txt")
docs = loader.load()

# Create embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vectorstore = FAISS.from_documents(docs, embeddings)

# Set up retriever
retriever = vectorstore.as_retriever()

# Conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

doc_prompt = PromptTemplate(
    template="""
You are an experienced medical doctor. You are provided with a reference medical guide (context). 
Using this context, diagnose the patient's symptoms, explain possible causes, and suggest appropriate treatments. 
Only give advice that is mentioned or inferred from the context, and do not give unrelated medical advice.

Context:
{context}

Patient Question:
{question}

Your response format:
Diagnosis: [Brief diagnosis based on the context]
Possible Causes: [List possible causes from the context]
Treatment: [Suggest remedies or medicines mentioned in the context]
Precaution: [Safety steps or when to seek emergency help]
""",
    input_variables=["context", "question"]
)








# Create ConversationalRetrievalChain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": doc_prompt}
)

# Chat loop
print("Welcome to DoctorBot. Ask your health question (type 'exit' to quit).")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Assistant: Thank you. Take care!")
        break
    response = qa_chain.invoke({"query": query})
    print("Assistant:", response["result"])
    print("--------------------------------------------------")
