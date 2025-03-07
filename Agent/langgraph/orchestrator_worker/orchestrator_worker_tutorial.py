from typing import Annotated,List,TypedDict
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langgraph.graph import StateGraph,START,END
from langgraph.constants import Send
import operator


class Section(BaseModel):
    name : str = Field(
        description="Name for this section of report"
    )
    description : str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section"
    )

class Sections(BaseModel):
    sections : List[Section] = Field(
        description="Sections of the report"
    )

load_dotenv()
llm = ChatOpenAI(
    model = "gpt-4o-mini"
)

planner = llm.with_structured_output(Sections)

## Graph
class State(TypedDict):
    topic : str  # report topic
    sections : list[Section] 
    completed_sections : Annotated[ # Annotated가 하는 일은 뭐지? 
        list, operator.add # operator가 뭐지
    ]
    final_report : str

# Worker state
class WorkerState(TypedDict):
    section : Section
    completed_sections : Annotated[list,operator.add] 

# Node
def orchestrator(state : State) :
    """Orchestrator that generates a plan for the report"""

    #Generate queries
    report_sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report"),
            HumanMessage(content = f"Here is the report topic: {state['topic']}")
        ]
    )
    return {"sections":report_sections.sections}

def llm_call(state:WorkerState):
    """Worker writes a section of the report"""
    # Generate Section
    section = llm.invoke(
        [
            SystemMessage(content="Write a report section"),
            HumanMessage(
                content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
            )
        ]
    )
    return {"completed_section" : section.content}

def synthesizer(state:State):
    """Synthesize full report from sections"""

    # List of completed sections
    completed_sections = state["completed_sections"]
    completed_report_sections = "\n\n----\n\n".join(completed_sections)

    return {"final_report":completed_report_sections}

def assign_workers(state: State):
    """Assign workers to each sections in the plan"""

    # section writing in parallel via Send() API
    return [Send("llm_call",{"section":s}) for s in state["sections"]]

## Workflow
orchestrator_worker_builder = StateGraph(State)

# Add the nodes
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_edge(START,"orchestrator")
orchestrator_worker_builder.add_conditional_edges("orchestrator",assign_workers,["llm_call"])
orchestrator_worker_builder.add_edge("llm_call","synthesizer")
orchestrator_worker_builder.add_edge("synthesizer",END)

orchestrator_worker = orchestrator_worker_builder.compile()

state = orchestrator_worker.invoke({"topic":"How to improve my english skill? Create a report on improving english skill"})
print("--------질    문--------")
print("How to improve my english skill? Create a report on improving english skill")
print("--------출력 결과--------")
print(state["sections"])
print("--------최종 결과--------")
print(state["final_report"])