from langchain_community.document_loaders import UnstructuredPDFLoader
from pdfminer import psparser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from IPython.display import display,Markdown

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
query = "students school dropout"
docs_faiss = faiss_db.similarity_search_with_score(query,k=3)

#Build RAG

retrieved_docs = "\n\n".join([doc.page_content for doc,_score in docs_faiss])
print(retrieved_docs) 
prompt = f"based on the content : {retrieved_docs} amswer the question : {query}"
llm = ChatOpenAI(
    openai_api_key = api_key,
    model="gpt-4o-mini",
    temperature=0.2)

display(Markdown(llm.invoke(prompt).content))
