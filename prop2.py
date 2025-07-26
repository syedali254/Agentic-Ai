from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import os

# ----------------------- API Setup -----------------------
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# ----------------------- TOOLS FOR PROPERTY AGENT -----------------------
properties = {
    "peshawar": {"location": "Hayatabad phase 1", "price": "2 crore", "size": "10 marla"},
    "lahore": {"location": "DHA phase 5", "price": "4 crore", "size": "1 kanal"},
    "karachi": {"location": "Bahria Town", "price": "3 crore", "size": "1 kanal"},
}

def find_property(city):
    city = city.lower()
    prop = properties.get(city)
    if prop:
        return f"Property in {city.title()}: {prop['location']}, Price: {prop['price']}, Size: {prop['size']}."
    else:
        return "Property not found."

def find_amenities(city):
    amenities_data = {
        "peshawar": "Nearby: Beaconhouse School, Shifa Hospital, Mall of Hayatabad",
        "lahore": "Nearby: LGS, National Hospital, Packages Mall",
        "karachi": "Nearby: Bay View Academy, Aga Khan Hospital, LuckyOne Mall"
    }
    return amenities_data.get(city.lower(), "No amenities data available.")

find_prop_tool = Tool(name="Find Property", func=find_property, description="Find property details in a city.")
amenities_tool = Tool(name="Amenities Finder", func=find_amenities, description="Find nearby amenities for a given city.")

# ----------------------- TOOLS FOR FINANCE AGENT -----------------------
def estimate_price(city_size):
    city_size = city_size.lower().replace(" ", "_")
    data = {
        "peshawar_10_marla": "Approx 2 crore",
        "lahore_1_kanal": "Approx 4 crore",
        "karachi_1_kanal": "Approx 3 crore"
    }
    return data.get(city_size, "No price data available.")

def loan_calculator(query):
    try:
        amount, rate, years = map(float, query.split(','))
        monthly_rate = (rate / 100) / 12
        months = years * 12
        emi = (amount * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
        return f"Monthly installment: {emi:.2f} PKR for {years} years at {rate}% interest."
    except Exception:
        return "Invalid input format. Use: 'amount, rate, years' (e.g., '20000000, 12, 20')."

def investment_advisor(budget_text):
    try:
        budget = float(budget_text)
        if budget < 20_000_000:
            return "Peshawar (Hayatabad Phase 1) is best for your budget."
        elif budget < 40_000_000:
            return "Karachi (Bahria Town) offers great value in your range."
        else:
            return "Lahore (DHA Phase 5) is the best long-term investment."
    except Exception:
        return "Enter a valid budget (e.g., '25000000')."

price_tool = Tool(name="Price Estimator", func=estimate_price, description="Estimate property price.")
loan_tool = Tool(name="Loan Calculator", func=loan_calculator, description="Calculate loan installments.")
investment_tool = Tool(name="Investment Advisor", func=investment_advisor, description="Suggest best city for investment.")

# ----------------------- RAG TOOL FOR FAQ AGENT -----------------------
loader = TextLoader("property_faq.txt")
docs = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
faiss_store = FAISS.from_documents(docs, embedding=embeddings)
retriever = faiss_store.as_retriever(search_kwargs={"k": 2})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

rag_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
def rag_query(question):
    return rag_chain.run(question)

rag_tool = Tool(name="Property FAQ Retriever", func=rag_query, description="Retrieve property-related answers.")

# ----------------------- SUB-AGENTS -----------------------
property_agent = initialize_agent(
    tools=[find_prop_tool, amenities_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

finance_agent = initialize_agent(
    tools=[price_tool, loan_tool, investment_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

faq_agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ----------------------- MANAGER AGENT (ROUTER) -----------------------
def route_query(query):
    q = query.lower()
    if any(word in q for word in ["loan", "emi", "price", "investment"]):
        return finance_agent.run(query)
    elif any(word in q for word in ["property", "amenities", "location"]):
        return property_agent.run(query)
    else:
        return faq_agent.run(query)

# ----------------------- TEST QUERIES -----------------------
print(route_query("Find property in Lahore and suggest nearby amenities."))
print(route_query("Estimate price for peshawar 10 marla."))
print(route_query("I have a budget of 25000000, which city is best for investment?"))
print(route_query("Tell me about Bahria Town Karachi."))
