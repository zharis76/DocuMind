from groq import Groq

def answer(question, results, api_key):
    client = Groq(api_key=api_key)
    
    context = "\n\n".join(
        f"[Page {r['page']}]\n{r['text']}"
        for r in results
    )
    
    system = """You are an assistant that answers questions based on the provided PDF document.

Instructions:
1. Answer using only the provided context.
2. Synthesize information from multiple relevant passages.
3. Do not invent facts or advice not supported by the document.
4. Cite the page number for every major claim.
5. Explain ideas in simple, clear language.
6. If the context does not contain enough information, say so.
7. Do not display similarity scores to the user.
8. Distinguish between the author's explicit statements and your own interpretations."""

    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content