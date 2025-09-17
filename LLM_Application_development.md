Chat With Your Data! The course delves into two main topics: (1) Retrieval Augmented Generation (RAG), a common LLM application that retrieves contextual documents from an external dataset, and (2) a guide to building a chatbot that responds to queries based on the content of your documents, rather than the information it has learned in training.

You’ll learn about:

Document Loading: Learn the fundamentals of data loading and discover over 80 unique loaders LangChain provides to access diverse data sources, including audio and video.

Document Splitting: Discover the best practices and considerations for splitting data.

Vector stores and embeddings: Dive into the concept of embeddings and explore vector store integrations within LangChain.

Retrieval: Grasp advanced techniques for accessing and indexing data in the vector store, enabling you to retrieve the most relevant information beyond semantic queries.

Question Answering: Build a one-pass question-answering solution.

Chat: Learn how to track and select pertinent information from conversations and data sources, as you build your own chatbot using LangChain.

Start building practical applications that allow you to interact with data using LangChain and LLMs.

-----

### Lo Scenario: Oltre la Conoscenza Pre-addestrata 🧠

Un Large Language Model, anche il più potente come GPT-4, opera con due limiti intrinseci. Primo, la sua conoscenza è **statica**, "congelata" all'ultimo giorno del suo addestramento; non sa nulla degli eventi recenti o dei dati privati della tua azienda. Secondo, in assenza di una fonte di verità a cui attingere, è soggetto ad **allucinazioni**, ovvero può generare risposte che suonano autorevoli ma sono fattualmente errate.

Il **Retrieval Augmented Generation (RAG)** è l'architettura progettata per superare questi limiti in modo elegante ed efficiente. Invece di tentare il costosissimo processo di ri-addestramento di un modello, ne "aumentiamo" la conoscenza al momento della domanda.

**L'analogia chiave:** Immagina un consulente a cui viene posta una domanda complessa. Invece di rispondere a memoria (rischiando di sbagliare o di non essere aggiornato), il consulente si ferma, si dirige verso uno schedario ben organizzato, estrae i documenti esatti relativi alla domanda, li legge attentamente e solo a quel punto formula una risposta basata *esclusivamente* su quelle fonti.

Il RAG automatizza questo processo: prima **recupera** (`Retrieval`) le informazioni pertinenti e poi le usa per **generare** (`Generation`) una risposta intelligente e ancorata ai fatti. Questo corso scompone l'intero processo in sei passaggi fondamentali, che ora analizzeremo con la massima profondità.

-----

### 1\. Document Loading: La Porta d'Accesso ai Dati 🚪

Il punto di partenza di ogni sistema RAG è l'ingestione dei dati. Non puoi interrogare ciò a cui il tuo sistema non può accedere. La fase di **Document Loading** consiste nel leggere i dati dalle loro fonti originali e caricarli in un formato standard che LangChain può manipolare, ovvero una lista di oggetti `Document`. Ogni oggetto `Document` contiene il testo (`page_content`) e un dizionario di informazioni contestuali (`metadata`).

La vera potenza di LangChain emerge qui, nella sua vasta e flessibile libreria di **`DocumentLoaders`**. Il testo menziona "oltre 80 loader unici", ed è questo che rende il framework così versatile. Non sei limitato a semplici file di testo (`.txt`) o PDF.

**Esempi di comandi e loader:**

  * **File Semplici:** Per un file PDF, il comando sarebbe semplice come:
    ```python
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader("documento_aziendale.pdf")
    documenti = loader.load()
    ```
  * **Contenuti Strutturati:** Per un file CSV, puoi usare `CSVLoader`, che caricherà ogni riga come un `Document` separato.
  * **Siti Web:** Per estrarre testo da una pagina web, si usa il `WebBaseLoader`:
    ```python
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader("https://blog.langchain.dev/...")
    documenti = loader.load()
    ```
  * **Contenuti Multimediali:** Qui l'innovazione è tangibile. Puoi usare `OpenAIWhisperParser` per trascrivere file **audio** e renderli interrogabili, o `YoutubeLoader` per scaricare e trascrivere il contenuto di un video.

Questa capacità di unificare fonti eterogenee (documenti, email, trascrizioni di riunioni, pagine del sito web aziendale) in un'unica base di conoscenza è il primo, fondamentale passo per costruire un sistema RAG robusto.

-----

### 2\. Document Splitting: L'Arte della Suddivisione Intelligente 쪼

Una volta caricato un documento, specialmente se lungo come un manuale tecnico o un libro, non puoi semplicemente passarlo interamente a un LLM. La fase di **Document Splitting** (o "chunking") è cruciale per due ragioni strategiche:

1.  **Limiti di Contesto:** Ogni LLM ha una "finestra di contesto" massima (es. 4k, 16k, 128k token). Passare un documento intero la supererebbe quasi certamente, risultando in un errore.
2.  **Efficienza del Recupero:** Quando un utente pone una domanda, vuoi trovare la porzione di testo *esatta* che contiene la risposta, non un intero capitolo generico. Frammenti più piccoli e focalizzati rendono il recupero molto più preciso e riducono il "rumore" passato all'LLM.

La sfida è "come" suddividere. Un taglio brutale può distruggere il significato. LangChain offre diverse strategie (`TextSplitter`) per farlo in modo intelligente.

#### Approfondimenti Tecnici sugli Splitter

  * **Misura della Lunghezza in Token:** Poiché i modelli LLM ragionano in **token**, la misura più accurata per la lunghezza di un chunk è il conteggio dei token. `TokenTextSplitter` è la scelta ideale, poiché si interfaccia con un "tokenizer" (come `tiktoken` di OpenAI) per garantire che i chunk non superino la dimensione desiderata.
    ```python
    from langchain.text_splitter import TokenTextSplitter
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documenti)
    ```
  * **`RecursiveCharacterTextSplitter`:** Questo è lo splitter più versatile e comunemente raccomandato. Invece di tagliare brutalmente, prova a suddividere il testo in modo gerarchico, cercando una lista di separatori in ordine. Di default, prova a dividere prima per doppi a capo (`\n\n` - paragrafi), poi per a capo singoli (`\n` - frasi), poi per spazi (`     `), e così via. Questo approccio massimizza la probabilità di mantenere intatte le unità semantiche del testo.
  * **`chunk_overlap` (Sovrapposizione):** Per evitare di perdere il contesto tra due chunk consecutivi, questo parametro fa sì che una parte del testo alla fine di un chunk venga ripetuta all'inizio di quello successivo. Questo crea una continuità che aiuta il modello a collegare concetti che si trovano a cavallo dei due frammenti. Un valore comune è il 10-20% della `chunk_size`.
  * **Splitting Contestuale (Context-Aware):** Le tecniche più avanzate, come lo **splitting semantico**, usano modelli di embedding per analizzare le frasi di un documento e inserire le interruzioni nei punti in cui il significato del testo cambia in modo più significativo, creando chunk tematicamente molto coerenti.
  * **Metadati:** È fondamentale che ogni chunk generato conservi i metadati del documento originale. LangChain lo fa automaticamente. Se un chunk proviene dalla pagina 42 di `report_2023.pdf`, i suoi metadati conterranno `{'source': 'report_2023.pdf', 'page': 42}`. Questo è vitale per citare le fonti e per tecniche di recupero avanzate.

-----

### 3\. Vector Stores and Embeddings: L'Archivio Semantico 📚

Qui avviene la "magia" che permette a un computer di comprendere il significato del testo.

  * **Embeddings:** Un embedding è una traduzione di un pezzo di testo in un linguaggio che il computer capisce: la matematica. È una rappresentazione numerica (un **vettore** di centinaia o migliaia di numeri) di quel testo. Modelli di embedding specializzati (es. `text-embedding-3-small` di OpenAI) generano questi vettori con una proprietà fondamentale: testi con significato simile avranno vettori matematicamente vicini nello spazio vettoriale.
  * **Vector Stores:** Un vector store (es. **FAISS**, **ChromaDB**, **Pinecone**) è un database ottimizzato per una singola, potentissima operazione: la **ricerca per similarità**.

#### La Matematica della Similarità

Quando un utente fa una domanda, anche questa viene trasformata in un vettore. Il vector store calcola la "distanza" tra il vettore della domanda ($V\_q$) e tutti i vettori dei chunk memorizzati. La metrica più comune è la **similarità cosenica**, legata al **prodotto scalare**. Un valore vicino a 1 significa che l'angolo ($\\theta$) tra i due vettori è vicino a 0, indicando una forte somiglianza semantica.

$$\text{similarity} = \cos(\theta) = \frac{V_q \cdot V_c}{\|V_q\| \|V_c\|}$$

#### Come Vengono Salvati i Documenti?

Il vector store non salva solo i vettori. Per ogni chunk, memorizza una coppia:

1.  **Il Vettore di Embedding:** Usato per la ricerca matematica.
2.  **Il Testo Originale del Chunk (e i metadati):** Ciò che viene effettivamente "recuperato" e passato all'LLM.

Il comando per creare e popolare un vector store in memoria con ChromaDB sarebbe:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings_model)
```

-----

### 4\. Retrieval: Oltre la Semplice Ricerca 🔍

La ricerca per similarità di base è potente, ma ha dei limiti. Per questo, si usano tecniche di recupero più sofisticate.

  * **Maximal Marginal Relevance (MMR):** Questo algoritmo risolve il problema della **diversità**. Invece di restituire i 5 chunk più simili, che potrebbero essere quasi identici tra loro, MMR seleziona un set di documenti che sia contemporaneamente **rilevante** per la domanda e **diverso** al suo interno, coprendo una gamma più ampia di informazioni. Si attiva semplicemente al momento della ricerca: `retriever = vectorstore.as_retriever(search_type="mmr")`.
  * **LLM-Aided Retrieval:** Qui, un LLM migliora il recupero stesso.
      * **Self-Querying Retriever:** L'LLM traduce una domanda in linguaggio naturale (es. "Cosa dicevano i report del 2023 sull'IA?") in una query strutturata che cerca sia il contenuto semantico sia un **filtro sui metadati** (`anno == 2023`).
      * **Multi-Query Retriever:** L'LLM riscrive la domanda dell'utente in diverse varianti per ampliare la ricerca e scoprire sfumature diverse.
  * **Compressione:** Un `ContextualCompressionRetriever` prima recupera molti documenti, poi usa un LLM per **estrarre e comprimere** solo le frasi pertinenti da ciascuno, ottimizzando lo spazio di contesto per l'LLM finale.

-----

### 5\. Question Answering: La Sintesi Intelligente della Risposta 💬

Questa è la fase in cui mettiamo tutto insieme. Dopo aver recuperato i documenti utili, come si genera la risposta? La catena di Question Answering (es. `RetrievalQA`) orchestra il processo: i chunk recuperati vengono inseriti in un **prompt** insieme alla domanda originale. Il prompt istruisce l'LLM a rispondere basandosi *esclusivamente* sul contesto fornito, eseguendo un compito di **lettura e sintesi** in tempo reale.

#### Approfondimento: `Map-Reduce`, `Refine`, `Map-Rerank`

Cosa succede se il numero di documenti recuperati è troppo grande per entrare in una singola finestra di contesto? Qui entrano in gioco strategie specifiche per gestire e aggregare le informazioni, specificate tramite il parametro `chain_type` nella catena `load_qa_chain`.

  * **`Map-Reduce`**

      * **Cosa fa:** È una tecnica a due fasi per parallelizzare l'elaborazione.
      * **Processo:**
        1.  **Fase Map:** Ogni documento recuperato viene passato *individualmente* all'LLM insieme alla stessa domanda. L'LLM genera una risposta parziale basata solo su quel singolo documento.
        2.  **Fase Reduce:** Tutte le risposte parziali generate nella fase Map vengono raccolte e passate *insieme* a un LLM finale, con l'istruzione di sintetizzarle in una risposta unica e coerente.
      * **Pro:** Può processare un numero enorme di documenti in parallelo, è veloce ed efficiente.
      * **Contro:** Può perdere il contesto globale, poiché ogni risposta parziale è generata in isolamento. La sintesi finale deve essere molto buona per unire i pezzi.
      * **Comando:** `chain = load_qa_chain(llm, chain_type="map_reduce")`

  * **`Refine`**

      * **Cosa fa:** È un processo iterativo e sequenziale che costruisce e "raffina" la risposta passo dopo passo.
      * **Processo:**
        1.  Il primo documento viene passato all'LLM per generare una risposta iniziale.
        2.  Il secondo documento viene passato all'LLM insieme alla domanda e alla *risposta precedente*, con l'istruzione di "raffinare" o migliorare la risposta con le nuove informazioni.
        3.  Questo processo continua per tutti i documenti, con la risposta che diventa progressivamente più completa.
      * **Pro:** Mantiene un forte contesto da un documento all'altro, ottimo per costruire risposte dettagliate e argomentate.
      * **Contro:** È intrinsecamente lento e non parallelizzabile. È anche più costoso a causa delle numerose chiamate sequenziali all'LLM.
      * **Comando:** `chain = load_qa_chain(llm, chain_type="refine")`

  * **`Map-Rerank`**

      * **Cosa fa:** È ideale per compiti di estrazione in cui si presume che la risposta si trovi in un singolo documento.
      * **Processo:**
        1.  **Fase Map:** Ogni documento recuperato viene passato individualmente all'LLM. Per ognuno, l'LLM deve fare due cose: generare una risposta e restituire un **punteggio di confidenza** sulla correttezza di quella risposta.
        2.  **Fase Rerank:** Il sistema semplicemente seleziona la risposta che ha ricevuto il punteggio di confidenza più alto e la restituisce come risposta finale.
      * **Pro:** È veloce, parallelizzabile e ottimo per trovare risposte fattuali e dirette.
      * **Contro:** Non è in grado di sintetizzare informazioni da più fonti.
      * **Comando:** `chain = load_qa_chain(llm, chain_type="map_rerank")`

-----

### 6\. Chat: La Sfida della Memoria Conversazionale 🗣️

Per trasformare il sistema in un **chatbot**, dobbiamo aggiungere la memoria.

#### Il Problema e la Soluzione `ConversationalRetrievalChain`

Un sistema RAG base è "stateless". Se l'utente chiede "E riguardo al secondo punto?", il retriever non sa a cosa si riferisce. La `ConversationalRetrievalChain` risolve questo problema in modo brillante:

1.  **Riceve la Nuova Domanda** ("E riguardo al secondo punto?").
2.  **Accede alla Cronologia della Chat**.
3.  **Condensa la Domanda:** Usa un LLM in un passaggio preliminare il cui unico compito è combinare la nuova domanda e la cronologia per creare una **nuova domanda autonoma** (es. "Quali sono i dettagli del secondo punto menzionato nel report finanziario?").
4.  **Esegue il Recupero:** È questa nuova domanda, completa e contestualizzata, che viene inviata al retriever.
5.  **Genera la Risposta Finale:** I documenti recuperati e la domanda originale vengono usati per generare la risposta finale.

Questo approccio a due LLM (uno per condensare la domanda, uno per rispondere) permette di creare un'esperienza di chat fluida e contestuale, trasformando dati statici in una fonte di conoscenza veramente interattiva.
