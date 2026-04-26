import os
import ollama
from transformers import pipeline, logging

# Solo muestra los errores producidos por BERTO, omite el resto de texto
logging.set_verbosity_error()

# Carga BERTO
clasificador_bert = pipeline(
    "zero-shot-classification", 
    model="Recognai/bert-base-spanish-wwm-cased-xnli" 
)

# Usa BERTO para comprobar que la pregunta está relacionada con informática
def validar_prompt(texto):
    etiquetas = ["literatura", "informática", "otros temas"]
    
    resultado = clasificador_bert(texto, candidate_labels=etiquetas)
    
    etiqueta_ganadora = resultado['labels'][0]
    confianza = resultado['scores'][0]
    
    print(f"DEBUG - BERT clasifica el prompt como: '{etiqueta_ganadora}' con certeza de {confianza:.2%}")
    
    if etiqueta_ganadora == "informática":
        return True
        
    return False


def chat(prompt):
    respuesta_completa = ""

    # Inicializa el contexto añadiendolo al historial con la etiqueta 'system'
    historial_mensajes = [{
        'role': 'system',
        'content': 'Te llamas Pingüevo y eres profesor de informática'
    }]

    # Valida el prompt de entrada
    if validar_prompt(prompt):


        # Añade el prompt al historial de mensajes
        historial_mensajes.append({'role': 'user', 'content': prompt})

        stream = ollama.chat(
            model='llama3.2:1b',
            messages=historial_mensajes,
            stream=True,
            options={
                'temperature': 0.2,
                'top_p': 0.5
            }
        )

        for chunk in stream:
            texto = chunk['message']['content']
            respuesta_completa += texto

        # Valida la respuesta del modelo
        if not (validar_prompt(respuesta_completa)):
            respuesta_completa = 'Lo siento, no te puedo ayudar con eso.'


        # Añade la respuesta al historial de mensajes
        historial_mensajes.append({'role': 'assistant', 'content': respuesta_completa})
    
    else:
        respuesta_completa = 'Lo siento, no te puedo ayudar con eso.'

    return respuesta_completa

print("\n" + "="*50)
print("¡Hola! Soy Pingüevo, tu profesor de informática.")
print("Escribe 'salir' para terminar la conversación.")
print("="*50 + "\n")

while True:
    usuario_input = input("Tú: ")
    
    if usuario_input.lower() == 'salir':
        print("Pingüevo: ¡Hasta la próxima clase!")
        break
        
    if not usuario_input.strip():
        continue
        
    respuesta = chat(usuario_input)
    print(f"\nPingüevo: {respuesta}\n")
    print("-" * 50)