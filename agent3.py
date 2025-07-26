from langchain.agents import initialize_agent, Tool, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
import os

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Tool 1 - Weather
def get_weather(city):
    return "Weather of Lahore is moderate hot with 33°C"

weather_tool = Tool(
    name="Weather Checker",
    func=get_weather,
    description="Get current weather for a given city."
)

# Tool 2 - Doctor Finder
doctors = {"Lahore": "Dr. Imran (Cardiologist), Fee: 2000 PKR"}
def find_cardiologist(city):
    doc = doctors.get(city)
    return doc if doc else "No cardiologist found in this city."
    
doctor_tool = Tool(
    name="Cardiologist Finder",
    func=find_cardiologist,
    description="Find best cardiologists in a given city."
)

# Tool 3 - Appointment Booking
appointments = []
import re
def book_appointment(text):
    doctor = re.search(r'Dr\.?\s+\w+', text)
    doctor = doctor.group(0) if doctor else "Unknown Doctor"
    day = re.search(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)', text, re.I)
    day = day.group(0) if day else "Unknown Day"
    name = re.search(r'My name is (\w+)', text)
    name = name.group(1) if name else "Unknown"
    phone = re.search(r'(\d{3,4}-\d{6,7}|\d{11})', text)
    phone = phone.group(0) if phone else "Unknown"

    return f"Appointment booked with {doctor} on {day} for {name} ({phone})."

appointment_tool = Tool(
    name="Appointment Booker",
    func=book_appointment,
    description="Book an appointment. The input will be a plain text instruction like 'Book appointment with Dr. Imran on Friday for Ali, phone 0321-1234567'."
)

# Initialize Agent
agent = initialize_agent(
    tools=[weather_tool, doctor_tool, appointment_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Query Example
response = agent.invoke({"input": "Who is the Doctor in Lahore and what is Weather of Lahore and book appointment on Friday with that doctor. My name is Ali and phone number is 0321-1234567."})
print(response)
