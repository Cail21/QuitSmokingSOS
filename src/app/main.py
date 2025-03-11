import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings
import time  # Aggiunto per l'effetto "typewriter"

# Soppressione dei warning di torch
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

# Utilizzo di PyMuPDFLoader invece di UnstructuredPDFLoader per risolvere problemi con le tabelle
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# Imposta la variabile d'ambiente per Protocol Buffers
# In questo modo si utilizza l'implementazione Python per evitare messaggi d'errore
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Definisce la cartella per la persistenza del database vettoriale (ChromaDB)
PERSIST_DIRECTORY = os.path.join("data", "vectors")

# Configurazione della pagina Streamlit
st.set_page_config(
    page_title="QuitSmokingSOS",
    page_icon="🚬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def typewriter(text: str, container: st.delta_generator.DeltaGenerator, delay: float = 0.005) -> None:
    """
    Effetto "macchina da scrivere": simula la scrittura del testo carattere per carattere.
    """
    output = ""
    for char in text:
        output += char
        container.markdown(output)
        time.sleep(delay)

# Estrae i nomi dei modelli dalle informazioni ricevute (models_info)
# Se l'attributo "models" non è presente, ritorna una tupla vuota
def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    logger.info("Extracting model names from models_info")
    try:
        if hasattr(models_info, "models"):
            model_names = tuple(model.model for model in models_info.models)
        else:
            model_names = tuple()
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()

# Crea un database vettoriale aggregando i blocchi (chunks) di tutti i file PDF caricati
# Per ciascun file PDF:
#   - Viene creata una directory temporanea
#   - Il file viene salvato e caricato tramite PyMuPDFLoader
#   - Il documento viene suddiviso in blocchi (chunks) tramite il text splitter
# Infine, vengono generati gli embeddings e creato il DB usando Chroma
def create_vector_db_multiple(file_uploads) -> Chroma:
    """
    Crea un vector DB aggregando i chunk di tutti i file PDF caricati.
    """
    logger.info("Creating vector DB from multiple uploads")
    all_chunks = []
    temp_dirs = []  # Per tenere traccia delle directory temporanee

    # Itera su ogni file PDF caricato
    for file_upload in file_uploads:
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        path = os.path.join(temp_dir, file_upload.name)
        
        # Salva il file nella directory temporanea
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
        logger.info(f"File {file_upload.name} saved in: {path}")
        
        # Carica il PDF tramite PyMuPDFLoader
        loader = PyMuPDFLoader(path)
        data = loader.load()
        
        # Suddivide il documento in blocchi (chunk)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=300)
        chunks = text_splitter.split_documents(data)
        all_chunks.extend(chunks)
        logger.info(f"Document {file_upload.name} split into {len(chunks)} chunks")
    
    # Genera gli embeddings per tutti i blocchi aggregati
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Crea un nome di collezione unico basato sui nomi dei file
    collection_name = f"pdf_collection_{hash(tuple(f.name for f in file_uploads))}"
    
    # Crea il database vettoriale con persistenza su disco
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=collection_name
    )
    logger.info("Vector DB created with persistent storage")
    
    # Rimuove le directory temporanee
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir)
        logger.info(f"Temporary directory {temp_dir} removed")
    
    return vector_db

# Elabora una domanda posta dall'utente utilizzando il database vettoriale e il modello di linguaggio selezionato.
# La funzione crea una catena (chain) che passa attraverso:
#   - Un retriever che genera query multiple
#   - Un prompt basato su un template
#   - Il modello LLM
#   - Un parser di output per formattare la risposta
# Ritorna la risposta generata e un placeholder per le referenze dei documenti
def process_question(question: str, vector_db: Chroma, selected_model: str) -> Tuple[str, str]:
    logger.info(f"Processing question: {question} using model: {selected_model}")
    # Inizializza il modello di linguaggio (LLM)
    llm = ChatOllama(model=selected_model)
    
    # Template del prompt per generare variazioni della query
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""  
Original question: {question}""",
    )
    
    # Configura il retriever che genera query multiple utilizzando il LLM
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )
    
    # Template del prompt per il processo di RAG (Retrieval Augmented Generation)
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
    prompt_template = ChatPromptTemplate.from_template(template)
    
    # Crea la chain che passa attraverso retriever, prompt, LLM e parser
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    # Placeholder per le referenze dei documenti (non disponibile in questa versione)
    doc_source = "Document references not available."
    return response, doc_source

# Estrae tutte le pagine di un file PDF come immagini.
# In input riceve il file PDF caricato e in output ritorna una lista di immagini per ciascuna pagina
@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

# Elimina il database vettoriale e rimuove dallo stato della sessione
# le informazioni relative al PDF (pagine, file, vector DB)
def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            vector_db.delete_collection()
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")

# Funzione principale che gestisce l'interfaccia utente con Streamlit:
#   - Gestione del caricamento (o scelta) del PDF (di esempio o upload multiplo)
#   - Creazione del database vettoriale e visualizzazione delle pagine PDF
#   - Interfaccia della chat per interagire con il modello LLM
def main() -> None:
    st.subheader("QuitSmokingSOS", divider="gray", anchor=False)
    
    # Ottiene le informazioni sui modelli disponibili tramite ollama
    models_info = ollama.list()
    available_models = extract_model_names(models_info)
    
    # Creazione del layout a due colonne: sinistra (col1) e destra (col2)
    col1, col2 = st.columns([1.5, 2])
    
    # Inizializzazione delle variabili di stato della sessione, se non già presenti
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False
    if "source" not in st.session_state:
        st.session_state["source"] = []  # Per salvare informazioni sulle referenze dei documenti
    
    # Se ci sono modelli disponibili, permette all'utente di selezionarne uno tramite una tendina
    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", 
            available_models,
            key="model_select"
        )
    else:
        selected_model = None
    
    # Permette di scegliere se utilizzare un PDF di esempio o caricare dei file
    use_sample = col1.toggle(
        "Use sample PDF (Collection of books about quitting smoking)", 
        key="sample_checkbox"
    )
    
    # Se si cambia la modalità (sample vs. upload), viene cancellato il vector DB esistente
    if use_sample != st.session_state.get("use_sample"):
        if st.session_state["vector_db"] is not None:
            st.session_state["vector_db"].delete_collection()
            st.session_state["vector_db"] = None
            st.session_state["pdf_pages"] = None
        st.session_state["use_sample"] = use_sample
    
    if use_sample:
        # Utilizza il PDF di esempio presente nel percorso specificato
        sample_path = "data/pdfs/sample/EasyWayStopSmoking.pdf"
        if os.path.exists(sample_path):
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing sample PDF..."):
                    loader = PyMuPDFLoader(file_path=sample_path)
                    data = loader.load()
                    # Suddivide il testo in blocchi
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=300)
                    chunks = text_splitter.split_documents(data)
                    st.session_state["vector_db"] = Chroma.from_documents(
                        documents=chunks,
                        embedding=OllamaEmbeddings(model="nomic-embed-text"),
                        persist_directory=PERSIST_DIRECTORY,
                        collection_name="sample_pdf"
                    )
                    # Estrae e salva le pagine del PDF come immagini
                    with pdfplumber.open(sample_path) as pdf:
                        st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
        else:
            st.error("Sample PDF file not found in the current directory.")
    else:
        # Consente il caricamento di più file PDF
        file_uploads = col1.file_uploader(
            "Upload PDF files ↓", 
            type="pdf", 
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        if file_uploads:
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing uploaded PDFs..."):
                    st.session_state["vector_db"] = create_vector_db_multiple(file_uploads)
                    st.session_state["file_upload"] = file_uploads
                    # Estrae le pagine da ciascun PDF e aggrega le immagini
                    pdf_images = []
                    for file in file_uploads:
                        with pdfplumber.open(file) as pdf:
                            pdf_images.extend([page.to_image().original for page in pdf.pages])
                    st.session_state["pdf_pages"] = pdf_images
    
    # Visualizza le pagine del PDF (se disponibili)
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=1000, 
            value=700, 
            step=50,
            key="zoom_slider"
        )
        with col1:
            with st.container():
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)
    
    # Pulsante per eliminare la collezione (il vector DB)
    delete_collection = col1.button(
        "Delete collection", 
        type="secondary",
        key="delete_button"
    )
    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])
    
    # Interfaccia della chat nella colonna di destra
    with col2:
        message_container = st.container()
    
        # Visualizza la cronologia dei messaggi della chat
        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    
        # Input per l'inserimento di un nuovo prompt
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                # Aggiunge il messaggio dell'utente alla cronologia
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)
    
                # Elabora la risposta e la visualizza
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None and selected_model:
                            response, doc_source = process_question(prompt, st.session_state["vector_db"], selected_model)
                            # Se presente, estrae l'eventuale "chain-of-thought" interno
                            if "<think>" in response and "</think>" in response:
                                start_idx = response.find("<think>")
                                end_idx = response.find("</think>")
                                chain_of_thought = response[start_idx+len("<think>"):end_idx].strip()
                                final_answer = response[end_idx+len("</think>"):].strip()
                            else:
                                chain_of_thought = ""
                                final_answer = response
                        else:
                            st.warning("Please upload a PDF file first.")
                            chain_of_thought = ""
                            final_answer = "No answer."
    
                    # Se disponibile, mostra la chain-of-thought in un expander
                    if chain_of_thought:
                        with st.expander("Show internal chain-of-thought"):
                            st.markdown(chain_of_thought)
    
                    # Visualizza la risposta finale con effetto typewriter
                    final_answer_container = st.empty()
                    typewriter(final_answer, final_answer_container)
    
                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append({"role": "assistant", "content": final_answer})
                    # Salva le informazioni sulla fonte (domanda, risposta e referenze dei documenti)
                    st.session_state["source"].append({
                        "question": prompt,
                        "answer": final_answer,
                        "document": doc_source
                    })
    
            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file or use the sample PDF to begin chat...")
    
if __name__ == "__main__":
    main()
