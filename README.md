# AI System Engineering     
## Angelo Caliolo - Davide Mancinelli

Tale progetto corrisponde a un chatbot LLM in cui sono utilizzati i RAG. Tale chatbot, chiamato QuitSmokingSOS, ha molteplici compiti che vertono sullo smettere di fumare e che saranno successivamente spiegati. Il vantaggio principale di tale sistema è la possibilità di utilizzo sia da parte di specialisti (operatori sanitari o psicologi) sia da parte di utenti senza alcuna conoscenza. 
E' possibile fornire in input sia prompt con dati biometrici sia prompt meno schematici, in cui sono espresse anche le proprie difficoltà e i propri miglioramenti da un punto di vista personale.


Per l'esecuzione di questo chatbot abbiamo utilizzato Ollama, Langchain e Streamlit.


# Struttura della cartella

QuitSmokingSOS/
├── src/                      # Codice sorgente
│   ├── app/                  # Streamlit
│   │   └── main.py           # Main dell'applicazione
├── data/                     # Dati archiviati
│   ├── pdfs/                 # PDF archiviati
│   │   └── sample/           # Sample PDF
│   └── vectors/              # Vector DB archiviati
└── run.py                    # Runner dell'applicazione



# Installazione con ambiente virtuale

Prerequisiti:

1. Ollama
2. Account Streamlit
3. Python da 3.8 a 3.11
4. Tesseract
5. Poppler


Guida all'installazione:

1. Installazione Ollama
    Per il download e l'installazione bisogna visitare https://ollama.ai
    Modelli richiesti:
    ```bash
        ollama pull deepseek-r1:1.5b    # O modello a scelta
        ollama pull nomic-embed-text
    ```

2. Clonazione della repository
    ```bash
        git clone
    ```
    Per accedere alla cartella del chatbot
    ```bash
        cd QuitSmokingSOS
    ```

3. Creazione dell'environment
    ```bash
        python -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    ```
    Nel caso in cui si stesse utilizzando Windows, il codice sarebbe questo:
    ```bash
        python -m venv venv
        source .\venv\Scripts\activate
        pip install -r requirements.txt
    ```
    I requisiti sono salvati all'interno del file testo requirements.txt
    Tali requisiti sono:
    ```txt
        ollama==0.4.4
        streamlit==1.40.0
        pdfplumber==0.11.4
        langchain==0.3.14
        langchain-core==0.3.29
        langchain-ollama==0.2.2
        langchain_community==0.3.14
        langchain_text_splitters==0.3.5
        unstructured>=0.16.12
        unstructured[all-docs]>=0.16.12
        onnx>=1.17.0
        protobuf==5.29.2
        chromadb>=0.4.22
        Pillow==10.4.0
        numpy==1.26.4
        pytest==7.4.4
        pytest-cov==4.1.0
        coverage==7.4.0
        pydantic==2.10.4
    ```

4. Esecuzione dell'applicazione
    Ollama deve essere in esecuzione.
    Per far partire l'esecuzione bisogna eseguire il comando:
    ```bash
        python run.py
    ```


# Installazione con Docker

Prerequisiti:

1. Docker


Guida all'installazione:

1. Aprire prompt dei comandi nella cartella del programma

2. Per avviare il contenitore eseguire il comando in modo tale da poter ancora lavorare sul bash
    ```bash
        docker compose up -d
    ```
    In alternativa, per mostrare anche il log 
    ```bash
        docker compose up
    ```

3. Per effettuare il pulling dei modelli:
    ```bash
        docker exec -it ollama ollama pull deepseek-r1:1.5b     # O modello a scelta
        docker exec -it ollama ollama pull nomic-embed-text
    ```

4. Per accedere al chatbot, aprire il browser e collegarsi a http://localhost:8501/
