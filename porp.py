from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
import os


api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)


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

find_prop_tool = Tool(
    name="Find Property",
    func=find_property,
    description="Find property details in a city. Input: city name (e.g., 'peshawar')."
)


def estimate_price(city_size):
    city_size = city_size.lower().replace(" ", "_")
    data = {
        "peshawar_10_marla": "Approx 2 crore",
        "lahore_1_kanal": "Approx 4 crore",
        "karachi_1_kanal": "Approx 3 crore"
    }
    return data.get(city_size, "No price data available.")

price_tool = Tool(
    name="Price Estimator",
    func=estimate_price,
    description="Estimate property price. Input format: 'city size' (e.g., 'peshawar 10 marla')."
)

def loan_calculator(query):
    try:
        amount, rate, years = map(float, query.split(','))
        monthly_rate = (rate / 100) / 12
        months = years * 12
        emi = (amount * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
        return f"Monthly installment: {emi:.2f} PKR for {years} years at {rate}% interest."
    except Exception:
        return "Invalid input format. Use: 'amount, rate, years' (e.g., '20000000, 12, 20')."

loan_tool = Tool(
    name="Loan Calculator",
    func=loan_calculator,
    description="Calculate loan installments. Input: 'amount, rate, years' (e.g., '20000000, 12, 20')."
)


def find_amenities(city):
    amenities_data = {
        "peshawar": "Nearby: Beaconhouse School, Shifa Hospital, Mall of Hayatabad",
        "lahore": "Nearby: LGS, National Hospital, Packages Mall",
        "karachi": "Nearby: Bay View Academy, Aga Khan Hospital, LuckyOne Mall"
    }
    return amenities_data.get(city.lower(), "No amenities data available.")

amenities_tool = Tool(
    name="Amenities Finder",
    func=find_amenities,
    description="Find nearby amenities for a given city."
)


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

investment_tool = Tool(
    name="Investment Advisor",
    func=investment_advisor,
    description="Suggest the best city for property investment based on your budget. Input: budget in PKR (e.g., '25000000')."
)


# Load knowledge base (can be FAQs or property rules)
from langchain_community.document_loaders import TextLoader

# Load the property knowledge base from file
loader = TextLoader("property_faq.txt")
docs = loader.load()

# Create FAISS Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
faiss_store = FAISS.from_documents(docs, embedding=embeddings)

retriever = faiss_store.as_retriever(search_kwargs={"k": 2})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

def rag_query(question):
    return rag_chain.run(question)

rag_tool = Tool(
    name="Property FAQ Retriever",
    func=rag_query,
    description="Retrieve property-related answers from FAQs or knowledge base. Input: question text."
)


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful real-estate agent AI. Use tools when needed to answer user's questions."),
    ("human", "{history}\n User:{input}")
])


agent = initialize_agent(
    tools=[find_prop_tool, price_tool, loan_tool, amenities_tool, investment_tool, rag_tool],
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    memory=memory
)


print(agent.run("Tell me about Loan and financing."))
#print(agent.run("Find property in Lahore and suggest nearby amenities."))
#print(agent.run("Estimate price for peshawar 10 marla."))
#print(agent.run("I have a budget of 25000000, which city is best for investment?"))
#print(agent.run("Tell me about Bahria Town Karachi."))
