### LangChain Basics

In LangChain for LLM Application Development, you will gain essential skills in expanding the use cases and capabilities of language models in application development using the LangChain framework.

In this course you will learn and get experience with the following topics:

Models, Prompts and Parsers: calling LLMs, providing prompts and parsing the response

Memories for LLMs: memories to store conversations and manage limited context space

Chains: creating sequences of operations

Question Answering over Documents: apply LLMs to your proprietary data and use case requirements

Agents: explore the powerful emerging development of LLM as reasoning agents.

This one-hour course, instructed by the creator of LangChain Harrison Chase as well as Andrew Ng will vastly expand the possibilities for leveraging powerful language models, where you can now create incredibly robust applications in a matter of hours.

### Models, Prompts and Parsers: Il Cuore dell'Interazione con gli LLM

Questo primo modulo del corso getta le fondamenta per interagire efficacemente con i modelli linguistici. Possiamo scomporlo in tre componenti chiave:

Models (Modelli):I modelli sono il cervello dell'operazione. Si tratta di modelli linguistici di grandi dimensioni, come GPT-3.5, GPT-4, Llama, o altri modelli open source, che sono stati addestrati su enormi quantità di testo per comprendere e generare linguaggio umano. LangChain non è un LLM, ma un framework che si "siede" sopra questi modelli, fornendo un'interfaccia standardizzata per interagire con essi. Questo significa che puoi, con relativa facilità, cambiare il modello sottostante alla tua applicazione senza dover riscrivere tutto il codice. La scelta del modello giusto dipende dalle tue esigenze in termini di prestazioni, costo e specializzazione.

Prompts (Suggerimenti): I prompt sono le istruzioni che forniamo all'LLM. L'arte del "prompt engineering" è fondamentale per ottenere i risultati desiderati. Un prompt ben formulato può fare la differenza tra una risposta generica e una precisa e utile. LangChain offre una serie di strumenti per gestire i prompt in modo efficiente. Ad esempio, le `PromptTemplate` consentono di creare modelli di prompt riutilizzabili con variabili. Questo è particolarmente utile in applicazioni complesse in cui i prompt possono diventare lunghi e dettagliati.

Parsers (Analizzatori Sintattici): I parser si occupano dell'output del modello. Spesso, non vogliamo solo una stringa di testo come risposta, ma dati strutturati, come un oggetto JSON, un elenco o un formato personalizzato. I parser di output di LangChain prendono l'output grezzo dell'LLM e lo trasformano nel formato desiderato. Questo è essenziale per integrare l'output dell'LLM in un'applicazione più ampia.

---

### Memories for LLMs: Dare un Contesto alle Conversazioni

Per impostazione predefinita, le interazioni con un LLM sono "stateless", ovvero ogni richiesta è indipendente dalle precedenti. Questo è un problema per applicazioni come i chatbot, dove è fondamentale ricordare il contesto della conversazione. Il modulo **Memory** di LangChain risolve questo problema, consentendo a un'applicazione di ricordare le interazioni passate e di utilizzarle per informare le risposte future. Esistono diversi tipi di memoria, come `ConversationBufferMemory` (che memorizza l'intera cronologia) o `ConversationSummaryMemory` (che riassume la conversazione per gestire contesti lunghi).

---

### Chains: Creare Sequenze di Operazioni

Le "Chains" (catene) sono al centro della filosofia di LangChain. Una catena è una sequenza di componenti collegate tra loro per eseguire compiti complessi. Le catene semplici come `LLMChain` combinano un modello e un prompt, ma la vera potenza risiede nell'orchestrare flussi di lavoro articolati.

#### Router Chains: Indirizzare le Richieste in Modo Intelligente 🧭

A volte, una singola catena non è sufficiente. Immagina di avere un assistente AI che deve rispondere a domande di fisica, ma anche a domande di biologia. Usare un unico prompt generico potrebbe non dare risultati ottimali. Le **Router Chains** risolvono questo problema agendo come un "centralino" intelligente.

Una Router Chain utilizza un LLM per analizzare l'input dell'utente e decidere quale, tra diverse catene a disposizione, è la più adatta a gestire quella specifica richiesta. Funziona con due componenti principali:

1.  Destination Chains (Catene di Destinazione): Sono le catene specializzate. Ognuna è progettata per un compito specifico, con un prompt e strumenti ottimizzati. Ad esempio, potresti avere una "PhysicsChain" con un prompt che incoraggia risposte basate su formule e principi fisici, e una "BiologyChain" con un prompt orientato alla terminologia biologica.
2.  Default Chain (Catena Predefinita): È la catena di "fallback", utilizzata quando la router chain non riesce a trovare una destinazione adeguata per l'input. Questo garantisce che l'applicazione possa sempre fornire una risposta, anche se generica.

Il processo è semplice: l'utente fa una domanda, la Router Chain la analizza, la instrada alla Destination Chain più appropriata (es. la PhysicsChain) e restituisce la sua risposta. Se nessuna corrisponde, entra in gioco la Default Chain. Questo approccio rende le applicazioni più efficienti, precise e scalabili.

---

### Question Answering over Documents: Sfruttare i Propri Dati

Questa è una delle applicazioni più potenti, nota come **Retrieval Augmented Generation (RAG)**. Permette agli LLM di rispondere a domande basate su dati privati o specifici di un dominio (documenti interni, articoli, ecc.), "ancorando" le risposte ai fatti e riducendo le allucinazioni.

Il processo RAG standard prevede il caricamento, la suddivisione (chunking), la creazione di rappresentazioni vettoriali (embeddings) e la memorizzazione in un database vettoriale (vector store). Quando arriva una domanda, il sistema recupera i documenti più pertinenti e li passa all'LLM insieme alla domanda per generare la risposta.

#### Gestire un Gran Numero di Documenti: `map_reduce`, `refine`, `map_rerank`

Cosa succede se la ricerca nel vector store restituisce un numero di documenti troppo grande per essere inserito nel contesto di una singola chiamata all'LLM? Qui entrano in gioco strategie specifiche per gestire e sintetizzare le informazioni:

* **`map_reduce`**: Questa tecnica prevede prima una fase di "mappatura", in cui ogni documento recuperato viene passato individualmente all'LLM con la stessa domanda per ottenere una risposta parziale. Successivamente, nella fase di "riduzione", tutte queste risposte parziali vengono combinate e sintetizzate in una risposta finale e coerente. È efficiente e parallelizzabile.
* **`refine`**: Questo approccio è iterativo. Il primo documento viene usato per generare una risposta iniziale. Poi, il secondo documento viene passato all'LLM insieme alla domanda e alla risposta precedente, chiedendo al modello di "raffinare" la risposta con le nuove informazioni. Il processo continua per tutti i documenti. È utile per costruire una risposta dettagliata e contestualizzata.
* **`map_rerank`**: In questo caso, ogni documento viene passato all'LLM per generare non solo una risposta, ma anche un punteggio di confidenza. L'applicazione seleziona poi la risposta con il punteggio più alto come risposta finale. È ideale quando si cerca la risposta più accurata e diretta proveniente da una singola fonte.

---

### Agents: Gli LLM come Motori di Ragionamento

Gli **"Agents"** rappresentano un'evoluzione significativa. Un agente non si limita a rispondere, ma usa un LLM come motore di ragionamento per decidere quali azioni intraprendere. Dotato di una serie di **"tools"** (strumenti) come API, calcolatrici o funzioni di ricerca, l'agente opera in un ciclo di "pensiero-azione-osservazione" per scomporre un problema complesso e risolverlo passo dopo passo, fino a raggiungere l'obiettivo finale.

---

### Valutazione dell'Applicazione LLM: Come Verificare la Qualità delle Risposte? 📊

Creare un'applicazione basata su LLM è solo metà del lavoro. Come facciamo a sapere se sta funzionando bene? La valutazione è un passo critico ma spesso trascurato. Verificare manualmente migliaia di risposte è impossibile, quindi abbiamo bisogno di un processo automatizzato.

L'idea di base è creare un **dataset di valutazione**, composto da una serie di domande di esempio e dalle relative **risposte di riferimento** ("ground truth"), ovvero le risposte che consideriamo corrette. Successivamente, facciamo girare la nostra applicazione su queste domande e confrontiamo le risposte generate (`prediction`) con quelle di riferimento (`ground truth`).

Strumenti come **LangSmith** di LangChain sono progettati specificamente per questo. Il processo di valutazione automatizzata si basa su diversi criteri, spesso affidando la valutazione a un LLM stesso:

* **Correttezza e Coerenza:** Si chiede a un LLM (come GPT-4) di agire come "giudice". Gli si fornisce la domanda, la risposta di riferimento e la risposta generata, e gli si chiede di valutare se la risposta generata è corretta, coerente e se contiene le stesse informazioni di quella di riferimento.
* **Assenza di Allucinazioni:** Si valuta se la risposta è "grounded", ovvero se è fedelmente basata sulle fonti (i documenti recuperati nel caso di RAG). L'LLM "giudice" controlla che il modello non abbia inventato informazioni non presenti nel contesto fornito.
* **Rilevanza:** Si verifica se la risposta è pertinente alla domanda posta dall'utente.
* **Tossicità e Sicurezza:** Si possono usare classificatori per verificare che l'output non sia dannoso, offensivo o inappropriato.

Usando LangSmith o altri framework di valutazione, puoi eseguire test automatici, monitorare il degrado delle performance nel tempo e confrontare diverse versioni dei tuoi prompt o della tua applicazione per vedere quale funziona meglio. Questo approccio basato sui dati è fondamentale per costruire applicazioni LLM affidabili e di alta qualità.
