import weaviate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from weaviate.classes.config import Configure, DataType, Property

client = weaviate.connect_to_local()

print(client.is_ready())


def createCollection(collectionName):
    if client.collections.exists(collectionName):
        print(f"WARNING: The collection '{collectionName}' already exists.")
        return

    client.collections.create(
        name=collectionName,
        vectorizer_config=Configure.Vectorizer.text2vec_ollama(
            api_endpoint="http://host.docker.internal:11434",
            model="nomic-embed-text",
        ),
        properties=[
            Property(
                name="content", data_type=DataType.TEXT, vectorize_property_name=False
            ),
            Property(name="origin", data_type=DataType.TEXT, skip_vectorization=True),
        ],
    )

    if client.collections.exists(collectionName):
        print("Collection created successfully.")
    else:
        print("ERROR: Collection not created.")


def deleteCollection(collectionName):
    if client.collections.exists(collectionName):
        client.collections.delete(collectionName)

    if not client.collections.exists(collectionName):
        print("Collection deleted successfully.")
    else:
        print("ERROR: Collection not deleted.")


def extractText(file_name):
    if file_name.lower().endswith(".pdf"):
        reader = PdfReader(file_name)
        full_text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                full_text += extracted_text + "\n"
        return full_text

    elif file_name.lower().endswith(".txt") or file_name.lower().endswith(".md"):
        with open(file_name, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format: {file_name}")


def insertData(collectionName, file_name):
    if not client.collections.exists(collectionName):
        print("ERROR: Collection does not exist.")
        return

    try:
        data = extractText(file_name)
    except FileNotFoundError:
        print(f"ERROR: File '{file_name}' not found.")
        return
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(data)

    objects_to_insert = [{"content": chunk, "origin": file_name} for chunk in chunks]

    collection = client.collections.get(collectionName)
    response = collection.data.insert_many(objects_to_insert)

    if response.has_errors:
        print("ERROR: Issues occurred during batch insertion.")
    else:
        print(f"SUCCESS: Inserted {len(objects_to_insert)} chunks from '{file_name}'.")


def main():
    while True:
        print("\n===== MENU =====")
        print("1. Crear coleccion")
        print("2. Borrar coleccion")
        print("3. Insertar datos")
        print("4. Salir")
        print("================")

        opcion = input("Selecciona una opcion: ").strip()

        if opcion == "1":
            nombre = input("Nombre de la coleccion: ").strip()
            createCollection(nombre)
        elif opcion == "2":
            nombre = input("Nombre de la coleccion: ").strip()
            deleteCollection(nombre)
        elif opcion == "3":
            nombre = input("Nombre de la coleccion: ").strip()
            archivo = input("Nombre del archivo: ").strip()
            insertData(nombre, archivo)
        elif opcion == "4":
            print("Saliendo...")
            break
        else:
            print("Opcion invalida.")


if __name__ == "__main__":
    main()
