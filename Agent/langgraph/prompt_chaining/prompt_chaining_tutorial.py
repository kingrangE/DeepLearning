from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,END,START
from IPython.display import Image,display

load_dotenv()

llm = ChatOpenAI(
    temperature = 0,
    model_name = "gpt-4o-mini"
)
# graph state
class State(TypedDict):
    topic : str
    joke : str
    improved_joke : str 
    final_joke : str

# nodes
def generate_jokes(state : State) :
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f'Write a short joke about {state['topic']}')
    return {"joke":msg.content}

def improve_jokes(state : State) :
    """Second LLM call to improve initial joke"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke":msg.content}

def final_jokes(state : State) :
    """Final LLM call for final polish"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state["improved_joke"]}")
    return {"final_joke":msg.content}

def check_punchline(state: State):
    """Gate function to check if the joke has a punchline"""
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass" 
    return "Fail"

## Build Workflow with langgraph
workflow = StateGraph(State)

## Add nodes (where llm use function at point)
workflow.add_node("generate_joke",generate_jokes)
workflow.add_node("improve_joke", improve_jokes)
workflow.add_node("polish_joke",final_jokes)

## Add edges to connect nodes
workflow.add_edge(START,"generate_joke")
## we have to check punchline. so we will use conditional_edges that devide edge 
workflow.add_conditional_edges("generate_joke",check_punchline,{"Pass":"improve_joke","Fail":END})
workflow.add_edge("improve_joke","polish_joke")
workflow.add_edge("polish_joke",END)

## compile
chain = workflow.compile()

## show workflow
# display(Image(chain.get_graph().draw_mermaid_png()))

##run
state = chain.invoke({"topic":"developer"})
print("Initial joke : ")
print(state["joke"])

if "improved_joke" in state :
    print("Improved joke")
    print(state["improved_joke"])
    
    print("Final joke")
    print(state["final_joke"])
else :
    print("check_punchline function caught Something that's not joke")