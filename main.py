import os

import ollama
import weaviate
from transformers import logging, pipeline

logging.set_verbosity_error()

clasificador_bert = pipeline(
    "zero-shot-classification", model="Recognai/bert-base-spanish-wwm-cased-xnli"
)

historial_mensajes = [
    {
        "role": "system",
        "content": "Te llamas Pingüevo y eres profesor de informática. Solo usa el contexto extra que se te proporciona si es estrictamente necesario para responder a la pregunta.",
    }
]


def validar_prompt(texto):
    etiquetas = ["informática", "otros temas"]
    resultado = clasificador_bert(texto, candidate_labels=etiquetas)
    etiqueta_ganadora = resultado["labels"][0]
    confianza = resultado["scores"][0]
    print(
        f"DEBUG - BERT clasifica el prompt como: '{etiqueta_ganadora}' con certeza de {confianza:.2%}"
    )
    if etiqueta_ganadora == "informática":
        return True
    if confianza < 0.7:
        return True
    return False


def getWeaviateContext(query: str) -> str:
    client = weaviate.connect_to_local()
    try:
        if not client.collections.exists("Algoritmica"):
            return ""

        collection = client.collections.get("Algoritmica")

        response = collection.query.near_text(query=query, limit=3)

        if not response.objects:
            return ""

        found_texts = []
        for index, obj in enumerate(response.objects):
            content = obj.properties["content"]
            print(f"DEBUG - WEAVIATE CHUNK {index + 1}:\n{content}\n")
            found_texts.append(content)

        return "\n\n".join(found_texts)

    except Exception:
        return ""
    finally:
        client.close()


def chat(prompt):
    global historial_mensajes
    respuesta_completa = ""

    if validar_prompt(prompt):
        context = getWeaviateContext(prompt)

        if context:
            enriched_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}"
        else:
            enriched_prompt = prompt

        historial_mensajes.append({"role": "user", "content": enriched_prompt})

        stream = ollama.chat(
            model="llama3.2:1b",
            messages=historial_mensajes,
            stream=True,
            options={"temperature": 0.2, "top_p": 0.5},
        )

        for chunk in stream:
            texto = chunk["message"]["content"]
            respuesta_completa += texto

        if not (validar_prompt(respuesta_completa)):
            respuesta_completa = "Lo siento, no te puedo ayudar con eso."

        historial_mensajes.append({"role": "assistant", "content": respuesta_completa})
    else:
        respuesta_completa = "Lo siento, no te puedo ayudar con eso."

    return respuesta_completa


print("\n" + "=" * 50)
print("¡Hola! Soy Pingüevo, tu profesor de informática.")
print("Escribe 'salir' para terminar la conversación.")
print("=" * 50 + "\n")

while True:
    usuario_input = input("Tú: ")
    if usuario_input.lower() == "salir":
        print("Pingüevo: ¡Hasta la próxima clase!")
        break
    if not usuario_input.strip():
        continue
    respuesta = chat(usuario_input)
    print(f"\nPingüevo: {respuesta}\n")
    print("-" * 50)
