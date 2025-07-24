from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


api_key=os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=api_key)

memory=ConversationBufferMemory()

prompt=ChatPromptTemplate.from_messages([

("system","Consider you are an real estate agent located in lahore dealing with lahore properties and respond using your existing  LLm knowldge to repond"),
("human","{history}\n User :{input}"),

])
conversation=ConversationChain(
llm=llm,
prompt=prompt,
memory=memory
)


print("Hello Welcome to LLM Real estate agent")
while(True):

    query=input()
    if query.lower() in ['quit','exit']:
        break
        print("Thanks for reaching us")

    result=conversation.run("query")
    print("Assistant",result)
    print("-------------------------------------------------")