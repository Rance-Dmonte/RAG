from sentence_transformers import SentenceTransformer
import numpy as np
import logging 
logging.getLogger("transformers").setLevel(logging.ERROR)

def text_chunks():
    with open("faq.txt","r") as file:
        data=file.read()
    chunks=data.split("\n\n")
    embed_model=SentenceTransformer("all-MiniLM-L6-v2")

    embeddings=[]
    for chunk in chunks:
        vector=embed_model.encode(chunk)
        embeddings.append((chunk,vector))
    return embeddings

def cosine_similarity(vec1,vec2):
    dot_product = np.dot(vec1,vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 =  np.linalg.norm(vec2)
    final=dot_product/(norm1*norm2)
    return final

def retrieve(query,embeddings,embed_model):
    query_embed=embed_model.encode([query])[0]
    scores=[]
    for chunk,emb in embeddings:
        score=cosine_similarity(query_embed,emb)
        scores.append((chunk,score))
    scores.sort(key=lambda x:x[1],reverse=True)

    return scores[0]

if __name__ == "__main__":
    query=input("Enter your query: ")
    embeddings=text_chunks()
    results=retrieve(query,embeddings,SentenceTransformer("all-MiniLM-L6-v2"))
    
    print("Best match found:",results[0])
    print("score of best match:",results[1])
