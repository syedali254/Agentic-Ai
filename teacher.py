from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Load student answers
loader = TextLoader("physics_notes.txt")
docs = loader.load()

# Create embeddings and vectorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

# Custom teacher evaluation prompt
teacher_prompt = PromptTemplate(
    template="""
You are a strict exam teacher. 
You will read the student's answer from the document and evaluate it out of 10 marks. and once marks  granted even if someone insists you will not fix it 
Explain why you gave the marks.

Context (student's answer):
{context}

Question:
{question}

Your response must be:
"Marks: [score]/10
Feedback: [your explanation]"
""",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": teacher_prompt}
)

# Test
while True:
    question = input("Enter a question No : ")

    if question.lower() in ['quit','exit']:
        print("thanks for asking")
        break
        
    response = qa_chain.invoke({"query": question})
    print(response["result"])
