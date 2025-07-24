from langchain.agents import initialize_agent, Tool, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
import os, requests

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Tool 1 - Weather
def get_weather(city):
    return "Weather of Lahore is Moderate hot with 33 C"

weather_tool = Tool(
    name="Weather Checker",
    func=get_weather,
    description="Get current weather for a given city."
)

# Tool 2 - Doctor Finder
doctors = {"Lahore": "Dr. Imran (Cardiologist), Fee: 2000 PKR"}
def find_cardiologist(city):
    doc=doctors.get(city)
    if doc:
        return doc
    return "No cardiologist found in this city."
    
doctor_tool = Tool(
    name="Cardiologist Finder",
    func=find_cardiologist,
    description="Find best cardiologists in a given city."
)

# Initialize Agent
agent = initialize_agent(
    tools=[weather_tool, doctor_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # decides which tool/function to call
    verbose=True
)

# Ask a question
response = agent.run("Who is the best cardiologist in Lahore and what is the weather there?")
print(response)


#you need to learn plan and execute agent 
#self ask and search
#conversational ReAct Description


#-----------------------------

# book appointment with dr imran on friday my name is Ali and phone number is 0321-1234567'

#you will call a book appointment function on tool