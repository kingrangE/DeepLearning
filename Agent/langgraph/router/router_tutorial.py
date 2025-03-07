from langgraph.graph import StateGraph,START,END
from typing_extensions import TypedDict, Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage

load_dotenv()
class Route(BaseModel):
    step : Literal["poem","story","joke"] = Field(
        None,description="The next step in the routing process"
    )

llm = ChatOpenAI(
    model = "gpt-4o-mini"
)
router = llm.with_structured_output(Route)
class State(TypedDict):
    topic : str
    route : str
    output : str

def call_llm_1(state:State) :
    msg = llm.invoke(f"Generate a joke about {state["topic"]}")
    return {"joke":msg.content}

def call_llm_2(state:State) :
    msg = llm.invoke(f"Generate a story about {state["topic"]}")
    return {"story":msg.content}

def call_llm_3(state:State) :
    msg = llm.invoke(f"Generate a poem about {state["topic"]}")
    return {"poem":msg.content}
def call_router(state : State):
    messages = [
        SystemMessage(
            content="Route the input to generate story,joke or poem on the user's request"
        ),
        HumanMessage(
            content=state["topic"]
        )
    ]
    route = router.invoke(messages)
    return {"route":route.step}

def route_decision(state : State):
    route = state["route"]
    if route=="joke":
        return call_llm_1
    elif route == "story":
        return call_llm_2
    elif route == "poem":
        return call_llm_3
    
## build graph
graph = StateGraph(State)

## add node
graph.add_node("router",call_router)
graph.add_node("gen_joke",call_llm_1)
graph.add_node("gen_story",call_llm_2)
graph.add_node("gen_poem",call_llm_3)

## connect node using edge
graph.add_edge(START,"router")
graph.add_conditional_edges("router",route_decision,{
    "joke":"gen_joke",
    "story":"gen_story",
    "poem":"gen_poem",
})
graph.add_edge("gen_joke",END)
graph.add_edge("gen_story",END)
graph.add_edge("gen_poem",END)

workflow = graph.compile()

state = workflow.invoke({"topic":"developer"})
print(f"route : {state["route"]}")
print(f"output : {state["output"]}")
print("========================================")
state = workflow.invoke({"topic":"sky"})
print(f"route : {state["route"]}")
print(f"output : {state["output"]}")
