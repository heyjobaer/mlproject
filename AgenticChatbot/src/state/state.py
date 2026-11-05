from typing import Annotated, TypedDict
from langgraph.graph import add_messages


class State(TypedDict):
  """
  Represent the structure of the state used in graph
  """
  messages: Annotated[list, add_messages]
  