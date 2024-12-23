from langchain_community.document_loaders import UnstructuredPDFLoader
from pdfminer import psparser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from IPython.display import display,Markdown,Image
from langchain.schema.output_parser import StrOutputParser
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,END

#Load Document
loader = UnstructuredPDFLoader("The Post-Inflation Economy that Could Be.pdf")
docs = loader.load()

#split Documents into chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
texts = text_splitter.split_documents(docs)

#create Embeddings

embeddings = OpenAIEmbeddings(openai_api_key=api_key,model="text-embedding-3-large")
#create Vector Database

faiss_db = FAISS.from_documents(texts,embeddings)

class AgentState(TypedDict):
  start : bool
  conversation : int
  question : str
  answer : str
  topic : bool
  documents : List
  recursion_limit : int
  memory : List
workflow = StateGraph(AgentState)

def greeting(state):
  print("Hello how may i help you")
  user_input = input("Query :- ")
  state["question"] = user_input
  state["conversation"] = 1
  state["memory"] = [user_input]
  return state
workflow.add_node("greetings",greeting)
workflow.set_entry_point("greetings")
app = workflow.compile()

def check_question(state):
  question = state["question"]
  system_prompt = """You are an advanced AI model responsible for determining if a given user query aligns with the specified topic. Use the following criteria:
                  Understand the topic context from the provided description.
                  Compare the user query with the topic scope, identifying relevance or deviation.
                  Respond with one of the following:
                  "Relevant" if the query is directly or indirectly related to the topic.
                  "Partially Relevant" if it loosely connects or has potential applicability.
                  "Irrelevant" if it doesn't relate to the topic at all.
                  Example Input:
                  Topic: Machine Learning for Data Science
                  User Query: "What is dynamic programming?"

                  Example Output:
                  "Relevant"

                  Do not provide explanations unless explicitly asked."""
  template = ChatPromptTemplate.from_messages([
      ("system",system_prompt),
      ("user question",{question})
  ])
  prompt = template.format_messages(question = question)
  llm = ChatOpenAI(
      openai_api_key = api_key,
      model="gpt-4o-mini",
      temperature=0.2)
  response_text = model.invoke(prompt)
  state["topic"] = response_text.content
  return state 

def topic_router(state):
  topic = state["topic"]
  if topic == True:
    return "no_topic"
  else:
    return "Off_topic"

def off_topic_response(state):
  print("Sorry i don't know about that")

def retrieval(state):
  memory = ",".join(state["memory"])
  docs_faiss = faiss_db.similarity_search_with_score(memory,k=3)
  retrieved_docs = "\n\n".join([doc.page_content for doc,_score in docs_faiss])
  state["documents"] = retrieved_docs
  return state

def generate_answer(state):
  retrieved_docs = state["documents"]
  query = state["question"]
  memory = state["memory"]
  system_prompt ="""You are an AI model tasked with providing precise,
   topic-relevant answers. Generate concise, accurate responses tailored to the user's query without unnecessary explanations unless requested.
   Stay focused and ensure alignment with the provided topic context."""
  llm = ChatOpenAI(
      openai_api_key = api_key,
      model="gpt-4o-mini",
      temperature=0.2)
  template = ChatPromptTemplate.from_messages([
      ("system",system_prompt),
      ("human",f"context:{retrieved_docs}"),
      (f"conversation so fare : {memory}")
      (f"query:{question}")])
  prompt = template.format_messages(question = query,retrieved_docs =retrieved_docs,memory = memory)
  response_text = llm.invoke(prompt)
  state["answer"] = response_text.content
  return state

workflow.add_node("check_question",check_question)
workflow.add_edge("greetings","check_question")
workflow.add_node("Off_topic_response",off_topic_response)
workflow.add_node("retrieval",retrieval)
workflow.add_node("generate_answer",generate_answer)
workflow.add_conditional_edges("check_question",topic_router,{"on_topic":"retrieval","off_topic":"Off_topic_response"})
workflow.add_edge("retrieval","generate_answer")
workflow.add_edge("generate_answer",END)
workflow.add_edge("Off_topic_response",END)
display(Image(app.get_graph(xray=True).draw_mermaid_png()))
