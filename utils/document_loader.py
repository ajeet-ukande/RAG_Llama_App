from langchain.document_loaders import PyPDFLoader


def load_documents(file_paths):
    all_documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)
    return all_documents
