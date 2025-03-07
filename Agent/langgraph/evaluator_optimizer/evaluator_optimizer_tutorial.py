from langgraph.graph import StateGraph,START,END
from typing_extensions import TypedDict, Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage

load_dotenv()

llm = ChatOpenAI(
    model = "gpt-4o-mini"
)
class Feedback(BaseModel):
    grade : Literal["funny","not funny"] = Field(
        description="Decide if the joke is funny or not. grade is strictly as possible"
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it"
    )

evaluator = llm.with_structured_output(Feedback)

class State(TypedDict):
    joke : str
    topic : str
    feedback : str
    funny_or_not : str

# Nodes
def llm_call_generator(state : State) :
    if state.get("feedback"): # agent replect feedback on next loop if agent got feedback
        msg = llm.invoke(
            f"Give me a joke about {state["topic"]} but take into account the feedback"
        )
    else : 
        msg = llm.invoke(
            f"Give me a joke about {state["topic"]}"
        )
    return {"joke" : msg.content}

def llm_call_evaluator(state: State):
    grade = evaluator.invoke(
        f"Grade below joke \n {state["joke"]} \n and If this joke is not funny, please leave feedback on how this joke could be improved."
    )
    return {"funny_or_not":grade.grade,"feedback":grade.feedback}

## conditional edge node 
def route_joke(state:State):
    """Route back to joke generator or end based upon feedback from the evaluator"""
    if state["funny_or_not"] == "funny" :
        return "Accepted"
    elif state["funny_or_not"] == "not funny" :
        print(f"Boring joke is evaluated by AI\njoke : {state["joke"]}\n")
        return "Rejected + Feedback"

## Build workflow
optimizer_builder = StateGraph(State)

## Add the nodes
optimizer_builder.add_node("llm_call_generator",llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator",llm_call_evaluator)

## Add the edges
optimizer_builder.add_edge(START,"llm_call_generator")
optimizer_builder.add_edge("llm_call_generator","llm_call_evaluator")
optimizer_builder.add_conditional_edges("llm_call_evaluator",route_joke,{
    "Accepted" : END,
    "Rejected + Feedback" : "llm_call_generator",
})

workflow = optimizer_builder.compile()

state = workflow.invoke({"topic" : "University"})

print(f"Funny Joke is evaluated by AI\njoke : {state["joke"]}")
print(f"Last Feedback is {state["feedback"]}")