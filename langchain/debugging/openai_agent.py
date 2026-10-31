from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()


def add(a: float, b: float) -> float:
  """Add two numbers.

  Args:
    a: first number
    b: second number
  """
  return a + b

def subtract(a: float, b: float) -> float:
  """Subtract b from a.

  Args:
    a: first number
    b: number to subtract from a
  """
  return a - b

def multiply(a: float, b: float) -> float:
  """Multiply two numbers.

  Args:
    a: first number
    b: second number
  """
  return a * b

def divide(a: float, b: float) -> float | str:
  """Divide a by b.

  Args:
    a: dividend
    b: divisor
  """
  if b == 0:
    return "Error: division by zero"
  return a / b


arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=200))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500))
tavily = TavilySearchResults()

tools = [arxiv, wiki, tavily, add, subtract, multiply, divide]

llm = ChatGroq(model="qwen/qwen3.8-27b")
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content=(
  "You are a helpful assistant. Always use the available tools to answer: "
  "use the math tools (add, subtract, multiply, divide) for any arithmetic, "
  "and wikipedia, arxiv or tavily for factual questions. Do not answer from memory."
))


class State(TypedDict):
  messages: Annotated[list[AnyMessage], add_messages]


def tool_call_llm(state: State):
  return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


def should_continue(state: State):
  """Route to the tools node if the LLM asked for tools, otherwise finish."""
  last_message = state["messages"][-1]
  if getattr(last_message, "tool_calls", None):
    return "tools"
  return END


def make_graph():
  builder = StateGraph(State)

  builder.add_node("tool_call_llm", tool_call_llm)
  builder.add_node("tools", ToolNode(tools))
  builder.add_edge(START, "tool_call_llm")
  builder.add_conditional_edges("tool_call_llm", should_continue, ["tools", END])
  builder.add_edge("tools", "tool_call_llm")

  # No checkpointer here: `langgraph dev` / LangGraph Studio adds its own
  return builder.compile()


graph = make_graph()
