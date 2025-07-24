from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

from langchain.chains import RetrievalQA,ConversationalRetrievalChain
import os 

api_key=os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=api_key)

#1 load the data
loader = TextLoader("physics_notes.txt")
docs =loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vectorstore=FAISS.from_documents(docs,embeddings)

memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)

retriever=vectorstore.as_retriever()
qa_chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever , memory=memory)


while True:
    question=input("Ask a Question: ")
    if question.lower() in ['quit','exit']:
        print("thanks for reaching")
        break
    response=qa_chain.invoke({"question":question})
    print(response['answer'])
    print("------------------------------------------------------------------")