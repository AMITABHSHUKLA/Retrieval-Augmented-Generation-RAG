
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")


def embed_text(text,tokenizer,model):
  inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
  with torch.no_grad():
    outputs = model(**inputs)
  embeddings = outputs.last_hidden_state
  embeddings = embeddings.mean(dim=1)
  #print("Mean embeddings [respresnting embeddings into a single unit rather than (batch_size, sequence_length, hidden_size)] :- ",embeddings)
  return embeddings


def document_embedding(documents):
  document_embeddings = []
  for doc in documents:
    doc_embedding = embed_text(doc,tokenizer,model)
    document_embeddings.append(doc_embedding)
  #putting document embedding into cpu for fassi indexing
  document_embeddings = torch.cat(document_embeddings).cpu().numpy
  return document_embeddings

doc_embeddings = document_embedding(documents)
#Intialize the faiss index.
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)


def retrieval(query,tokenizer,model,index,documents,tok_k = 3):
  query_embedding = embed_text(query,tokenizer,model)
  distance,indices = index.search(query_embedding,k=tok_k)
  print("Distance:- ",distance,"\n","Indices :- ",indices)
  return [documents[i] for i in indices[0]]

gen_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gen_model = AutoModelForCausalLM.from_pretrained("gpt2")

def genrated_text(retrieved_docs,query,gen_modal,gen_tokenizer):
  gen_tokenizer.pad_token = gen_tokenizer.eos_token
  context = " ".join(retrieved_docs)
  input_text = f"context: {context} \n question: {query} \n answer:"
  inputs = tokenizer(input_text,return_tensor = "pt",padding = True,truncation = True)
  input_ids = inputs[input_ids]
  attention_mask = [input_ids != tokenizer.pad_token_id].long()
  outputs = gen_model.generate(input_ids,attention_mask= attention_mask,max_length = 100,pad_token_id = tokenizer.eos_token_id)
  generated_text = tokenizer.decode(outputs[0],skip_special_tokens = True)
  return generated_text

def rag(query,retrieval_tokenizer,retrieval_model,retrieval_index,gen_model,gen_tokenizer,documents,top_k):
  retrieved_docs = retrieval(query,tokenizer,model,index,documents,tok_k = 3)
  context = " ".join(retrieved_docs)
  generated_answer = genrated_text(retrieved_docs,query,gen_modal,gen_tokenizer)
  return generated_answer
