from langgraph.graph import StateGraph,START,END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model = "gpt-4o-mini"
)

class State(TypedDict):
    topic : str
    joke : str 
    poem : str 
    story : str 
    combined_output : str

def call_llm_1(state:State) :
    msg = llm.invoke(f"Generate a joke about {state["topic"]}")
    return {"joke":msg.content}

def call_llm_2(state:State) :
    msg = llm.invoke(f"Generate a story about {state["topic"]}")
    return {"story":msg.content}
def call_llm_3(state:State) :
    msg = llm.invoke(f"Generate a poem about {state["topic"]}")
    return {"poem":msg.content}
def aggregator(state:State) :
    combined = f'Here\'s a joke, story, poem about {state["topic"]}'
    combined += f'\njoke : {state["joke"]}'
    combined += f'\nstory : {state["story"]}'
    combined += f'\npoem : {state["poem"]}'

    return {"combined_output":combined}

## build graph
graph = StateGraph(State)

## add node
graph.add_node("gen_joke",call_llm_1)
graph.add_node("gen_story",call_llm_2)
graph.add_node("gen_poem",call_llm_3)
graph.add_node("aggregate",aggregator)

## connect node using edge
graph.add_edge(START,"gen_joke")
graph.add_edge(START,"gen_story")
graph.add_edge(START,"gen_poem")
graph.add_edge("gen_joke","aggregate")
graph.add_edge("gen_story","aggregate")
graph.add_edge("gen_poem","aggregate")
graph.add_edge("aggregate",END)

workflow = graph.compile()

state = workflow.invoke({"topic":"developer"})
print(state["combined_output"])