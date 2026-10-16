"""The C2 workflow as a LangGraph graph (Step 3: one round, rule-based, no LLM)."""
from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import C2State


def build_graph():
    g = StateGraph(C2State)

    g.add_node("load_input", nodes.load_input)
    g.add_node("profile_task", nodes.profile_task)
    g.add_node("agent1_select", nodes.agent1_select)
    g.add_node("agent2_hpo", nodes.agent2_hpo)
    g.add_node("evaluate", nodes.evaluate)
    g.add_node("agent1_diagnose", nodes.agent1_diagnose)
    g.add_node("agent3_select", nodes.agent3_select)
    g.add_node("package_output", nodes.package_output)

    g.add_edge(START, "load_input")
    g.add_edge("load_input", "profile_task")
    g.add_edge("profile_task", "agent1_select")
    g.add_edge("agent1_select", "agent2_hpo")
    g.add_edge("agent2_hpo", "evaluate")
    g.add_edge("evaluate", "agent1_diagnose")
    # Step 4 replaces this edge with a conditional edge:
    #   agent1_diagnose -> (next round) agent2_hpo   OR   (stop) agent3_select
    g.add_edge("agent1_diagnose", "agent3_select")
    g.add_edge("agent3_select", "package_output")
    g.add_edge("package_output", END)

    return g.compile()
