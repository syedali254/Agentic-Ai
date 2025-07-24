from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

api_key=os.getenv("GOOGLE_API_KEY")
llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=api_key)

memory=ConversationBufferMemory(return_messages=True)

promp=ChatPromptTemplate.from_messages([
("system","consider you are a doctor dealing with a patient but make sure to repond briefly in 3 lines"),
("human","{history}\n User:{input}"),
])

conversation=ConversationChain(llm=llm , memory=memory , prompt=promp)

print("Welcome whats your problem?")
while(True):
    quest=input()
    if quest.lower() in ['quit','exit']:
        print(f"Thanks for  reaching us ")
        break
    result=conversation.run(quest)
    print("Doctor :",result)
    print("---------------------------------------------")




