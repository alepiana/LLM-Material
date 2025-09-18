### LangGraph - Course Outline

LangChain, a popular open source framework for building LLM applications, recently introduced LangGraph. This extension allows developers to create highly controllable agents.

In this course you will learn to build an agent from scratch using Python and an LLM, and then you will rebuild it using LangGraph, learning about  its components and how to combine them to build flow-based applications.

Additionally, you will learn about agentic search, which returns multiple answers in an agent-friendly format, enhancing the agent’s built-in knowledge. This course will show you how to use agentic search in your applications to provide better data for agents to enhance their output.

In detail:

Build an agent from scratch, and understand the division of tasks between the LLM and the code around the LLM.
Implement the agent you built using LangGraph.
Learn how agentic search retrieves multiple answers in a predictable format, unlike traditional search engines that return links.
Implement persistence in agents, enabling state management across multiple threads, conversation switching, and the ability to reload previous states.
Incorporate human-in-the-loop into agent systems.
Develop an agent for essay writing, replicating the workflow of a researcher working on this task.

## Introduzione: La Ricerca del Controllo e della Trasparenza negli Agenti AI

L'avvento di framework come LangChain ha democratizzato lo sviluppo di applicazioni basate su LLM, permettendo di creare prototipi funzionanti in tempi record. Tuttavia, man mano che queste applicazioni si sono evolute da semplici catene di prompt a complessi agenti autonomi, sono emersi dei limiti strutturali. Le prime architetture di agenti, basate su un loop imperativo (come il classico `AgentExecutor` di LangChain), sebbene potenti, operavano spesso come "scatole nere". LangChain, con la sua LangChain Expression Language (LCEL), è eccezionale per costruire "catene" di operazioni. Queste catene sono potenti, ma per loro natura sono grafi aciclici diretti (DAG). Questo significa che il flusso di dati va sempre in una sola direzione, dall'inizio alla fine, senza mai tornare indietro, e risulta essere un limite quando si vogliono costruire sistemi più intelligenti e autonomi, come gli agenti.

Le sfide erano significative:

  * **Fragilità:** Il "pensiero" dell'agente (la catena di `Thought`, `Action`, `Observation`) era un blocco di testo monolitico. Un singolo errore di formattazione da parte dell'LLM poteva far fallire l'intero processo.
  * **Mancanza di Controllo:** Intervenire a metà di un'esecuzione, correggere una decisione sbagliata o guidare l'agente era quasi impossibile senza interrompere e riavviare tutto.
  * **Difficoltà di Debug:** Capire *perché* un agente si fosse bloccato in un loop o avesse preso una decisione illogica richiedeva un'analisi complessa di lunghi e intricati log di testo.

**LangGraph** è nato come risposta diretta a queste sfide. Rappresenta un cambio di paradigma fondamentale: si passa da un **loop imperativo** scritto in Python a un **grafo di esecuzione dichiarativo e stateful**. Invece di dire all'agente *come* eseguire il loop, si definisce una mappa dei possibili stati e transizioni, e si lascia che un motore di esecuzione gestisca il flusso. 

Per capire ancora meglio la differenza usiamo una metafora.

**LangChain è una Catena di Montaggio** 🚂: le componenti (LLMs, Tools) sono le singole stazioni di lavoro lungo la linea, mentre LCEL è il nastro trasportatore che collega le stazioni in una sequenza fissa e predeterminata. La catena risulta efficiente per compiti lineari. Ma cosa succede se un bullone è avvitato male? Il pezzo continua ad andare avanti. Non può tornare indietro alla stazione precedente per una correzione. 

**LangGraph è un'Officina di Artigiani Esperti** 🛠️: i nodi sono gli artigiani specializzati, lo stato è il pezzo su cui tutti lavorano, contiene il prodotto e un foglio di lavoro con le specifiche e i risultati dei test, ed infine gli archi condizionali (edges) sono le decisioni del capofficina (il grafo stesso). Dopo che un artigiano ha finito, il capofficina guarda il pezzo e decide cosa fare successivamente. In questa officina, il processo non è rigido. Il team può collaborare, tornare sui propri passi e adattare il flusso di lavoro in tempo reale per produrre un risultato migliore e più complesso. LangGraph ti dà gli strumenti per costruire proprio questo tipo di "officina intelligente" per i tuoi agenti AI.

Questo corso ti accompagna in questo viaggio, dalla comprensione dei limiti del vecchio approccio alla padronanza delle infinite possibilità del nuovo.

-----

## Parte 1: Le Fondamenta - Sezionare un Agente ReAct da Zero

Prima di poter costruire un grattacielo, è necessario conoscere i mattoni. La costruzione di un agente **ReAct (Reason + Act)** da zero è un esercizio didattico cruciale perché rivela la chiara divisione dei compiti tra l'intelligenza artificiale e il codice convenzionale.

### Il Cervello e le Mani: Una Collaborazione Simbiotica

  * **LLM (il Cervello 🧠):** Il Large Language Model è il puro motore di **ragionamento**. Il suo unico ruolo è quello di pensare. All'interno del prompt, gli viene fornito un contesto (la domanda dell'utente, la cronologia della chat, gli strumenti disponibili) e una "memoria di lavoro" chiamata **`agent_scratchpad`**. Questa memoria contiene la cronologia dei suoi pensieri e delle azioni precedenti. Il suo compito è analizzare tutto questo e produrre un testo che segua uno schema preciso, indicando il suo pensiero (`Thought`) e la sua prossima azione (`Action` e `Action Input`).

  * **Codice (le Mani e gli Occhi 🛠️):** Il codice Python che avvolge l'LLM è il sistema di esecuzione. Non ragiona, ma agisce. Il suo ciclo di vita è:

    1.  **Invocare l'LLM:** Chiama il "cervello" per ottenere la prossima mossa.
    2.  **Interpretare l'Output:** Analizza il testo generato dall'LLM per estrarre l'azione da compiere (es. `ToolCall('TavilySearchResults', {'query': '...'})`).
    3.  **Eseguire l'Azione:** Chiama la funzione Python o l'API corrispondente allo strumento scelto.
    4.  **Osservare il Risultato:** Raccoglie l'output dello strumento.
    5.  **Aggiornare lo Scratchpad:** Formatta l'azione e l'osservazione in un testo e lo aggiunge allo `agent_scratchpad`, arricchendo il contesto per il prossimo ciclo di ragionamento.

### L'Anatomia del `agent_scratchpad`

Il cuore dell'agente ReAct è questo "monologo interiore" dell'agente. Dopo un paio di cicli, potrebbe apparire così:

```
Thought: L'utente vuole sapere le ultime notizie sull'IA. Dovrei usare il mio strumento di ricerca web.
Action: TavilySearchResults
Action Input: {"query": "ultime notizie intelligenza artificiale"}
Observation: L'azienda 'InnovateAI' ha annunciato un nuovo modello chiamato 'Phoenix-1'.
Thought: Interessante. Ora l'utente potrebbe volere più dettagli su questo modello. Devo cercare 'Phoenix-1 AI model'.
Action: TavilySearchResults
Action Input: {"query": "Phoenix-1 AI model"}
```

Costruire questo manualmente ti fa capire quanto sia meticoloso il processo e quanto sia facile che si rompa. Questo dolore è la motivazione perfetta per LangGraph.

-----

## Parte 2: Ricostruire con LangGraph - Il Potere del Grafo Stateful

LangGraph prende la logica del loop ReAct e la mappa su una struttura a grafo. Invece di un `while` loop, si definisce un flusso di stati e transizioni.

### I Componenti Fondamentali

1.  **Lo Stato (State): La Memoria Centrale**
    Lo **stato** è il concetto più importante. È un oggetto Python che contiene *tutti* i dati di una sessione. È la singola fonte di verità. La sua definizione è flessibile e dipende dall'applicazione.

      * **Definizione:** Si usa quasi sempre un `TypedDict` per avere type safety.
        ```python
        from typing import TypedDict, Annotated, Sequence
        from langchain_core.messages import BaseMessage
        import operator

        # Stato per un chatbot semplice
        class BasicChatState(TypedDict):
            input: str
            chat_history: Annotated[Sequence[BaseMessage], operator.add]
            response: str

        # Stato per un agente di ricerca complesso
        class ResearchAgentState(TypedDict):
            task: str
            plan: list[str]
            retrieved_docs: Annotated[list, operator.add]
            draft: str
            critique: str
        ```

    L'uso di `Annotated[..., operator.add]` è un'istruzione per LangGraph: quando un nodo restituisce un valore per questo campo, non sostituire il valore esistente, ma **aggiungilo** (concatenando liste o stringhe).

2.  **I Nodi (Nodes): Le Unità di Lavoro**
    I nodi sono le funzioni che compongono il grafo. Ricevono lo stato corrente e restituiscono un dizionario con gli aggiornamenti. La distinzione di colore che hai notato in LangSmith è una visualizzazione della loro funzione:

      * **Nodi di Ragionamento (Blu/Viola):** Contengono la logica che invoca un LLM. Qui risiedono i **prompt**.
        ```python
        def agent_node(state: ResearchAgentState):
            # Costruisce il prompt usando i dati dallo stato
            prompt = create_agent_prompt(state['task'], state['retrieved_docs'])
            # Invoca l'LLM
            result = prompt | model
            # Restituisce l'aggiornamento allo stato
            return {"plan": result.plan}
        ```
      * **Nodi di Azione (Arancione/Giallo):** Eseguono strumenti. Non usano prompt.
        ```python
        def search_node(state: ResearchAgentState):
            # Legge il piano dallo stato
            topic_to_search = state['plan'][-1]
            # Esegue lo strumento
            results = tavily_tool.invoke({"query": topic_to_search})
            # Restituisce l'aggiornamento allo stato
            return {"retrieved_docs": results}
        ```

3.  **Gli Archi (Edges): Il Flusso Logico**
    Gli archi collegano i nodi. Gli **archi condizionali** sono il motore della logica dell'agente. Sono funzioni che ispezionano lo stato e decidono quale nodo eseguire dopo.

    ```python
    def should_continue(state: ResearchAgentState):
        # Se c'è una bozza finale, termina
        if state.get('draft') and not state.get('critique'):
            return "end"
        # Altrimenti, continua a ricercare
        else:
            return "search_node"

    # Nel grafo...
    graph.add_conditional_edges("agent_node", should_continue, {"end": END, "search_node": "search_node"})
    ```

-----

## Parte 3: Migliorare l'Input - La Precisione della Ricerca Agentica

Un agente che opera nel mondo reale ha bisogno di dati freschi e affidabili. Affidarsi solo alla conoscenza interna dell'LLM porta a risposte obsolete e a un'alta probabilità di allucinazioni.

### Perché la Ricerca Standard è Inadeguata

Quando un LLM riceve una lista di link, affronta diversi problemi:

  * **Costo Computazionale:** Deve "visitare" ogni pagina, un'operazione lenta.
  * **Rumore:** Le pagine web sono piene di pubblicità, menu, cookie banner e contenuti irrilevanti. Estrarre il segnale dal rumore è difficile.
  * **Contesto Rotto:** L'informazione potrebbe essere sparsa su più pagine, rendendo difficile la sintesi.

La **Ricerca Agentica**, implementata da strumenti come **Tavily**, risolve questi problemi. Esegue una pre-elaborazione per conto dell'agente, agendo come un assistente di ricerca. Quando l'agente chiede "Qual è l'impatto di Phoenix-1 sul mercato AI?", Tavily non restituisce link, ma un JSON strutturato con una risposta diretta, citazioni, e fonti, già pulito e pronto per essere inserito nel prompt dell'LLM. Questo fornisce dati fattuali e contestuali, riducendo drasticamente le allucinazioni e permettendo all'agente di avere conversazioni informate su eventi accaduti pochi minuti prima.

-----

## Parte 4: Rendere l'Agente Robusto, Persistente e Interattivo

Questa sezione si concentra sulle funzionalità che elevano un prototipo a un'applicazione affidabile. Il collante di tutto è la **persistenza**.

### Persistenza con il Checkpointer 💾

Collegando un **Checkpointer** (`SqliteSaver` per iniziare, `PostgresSaver` per la produzione) al grafo, ogni modifica allo stato viene salvata in un database.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)
```

Questo singolo componente abilita un mondo di funzionalità:

  * **Memoria e Thread:** Ogni conversazione viene salvata con un `thread_id`. Puoi gestire migliaia di conversazioni simultanee. Invocando il grafo con `configurable={"thread_id": "user-123"}`, LangGraph carica automaticamente l'ultimo stato di quella conversazione, la fa avanzare di un passo e salva il nuovo stato.

### Streaming in Tempo Reale

Per un'esperienza utente moderna, l'attesa non è un'opzione. Il metodo `.stream()` è la soluzione.

```python
# Itera sui risultati dei nodi man mano che vengono completati
for step in graph.stream(input_data, configurable={"thread_id": "user-123"}):
    print(step)
```

Per lo streaming dei singoli token dall'LLM, si usa `.astream_events()`, che richiede un'architettura asincrona e un `AsyncSqliteSaver`. Questo permette di replicare l'effetto di "scrittura in diretta" di interfacce come ChatGPT.

### Human-in-the-Loop e il Potere del Time Travel 🔄

Questa è la caratteristica più rivoluzionaria per il controllo degli agenti. La persistenza ti permette di diventare un partecipante attivo nel ragionamento dell'agente.

  * **Interruzione per Approvazione:** Puoi configurare il grafo perché si fermi prima di un nodo critico (es. `graph.add_edge('agent_node', 'human_approval_node')`). L'esecuzione si blocca, e puoi ispezionare lo stato e decidere se continuare o meno.
  * **Time Travel e Modifica dello Stato:** Immagina questo scenario: un agente di ricerca decide erroneamente di concentrarsi su un argomento irrilevante.
    1.  **Osservi l'errore:** Noti nello stream che lo strumento di ricerca viene chiamato con una query sbagliata.
    2.  **Viaggi nel Tempo:** Recuperi lo stato *prima* della decisione sbagliata.
        ```python
        # Recupera la lista di tutti gli stati salvati per il thread
        history = graph.get_state_history(configurable={"thread_id": "research-789"})
        # Prendi lo stato precedente a quello corrente
        previous_state = history[-2]
        ```
    3.  **Modifichi lo Stato:** Intervieni direttamente sull'oggetto Python.
        ```python
        # Correggi il piano dell'agente
        previous_state.values['plan'] = ["Focus on relevant topic A", "Ignore topic B"]
        ```
    4.  **Aggiorni la Storia:** Salvi questa versione corretta dello stato, sovrascrivendo la cronologia da quel punto in poi.
        ```python
        graph.update_state(
            configurable={"thread_id": "research-789", "thread_ts": previous_state.id},
            value=previous_state.values
        )
        ```
    5.  **Riavvi l'Esecuzione:** La prossima volta che invochi il grafo per questo thread, partirà dallo stato corretto. Le sue decisioni successive **rifletteranno immediatamente la tua correzione**, cambiando drasticamente il corso dell'esecuzione. Questa non è solo memoria; è memoria interattiva e modificabile.

-----

## Parte 5: Architetture Avanzate e Ingegneria dei Flussi

LangGraph eccelle nell'orchestrare non solo un singolo agente, ma "società di agenti". Questa è l'**Ingegneria dei Flussi**.

  * **Supervisor e Agenti Specializzati:** Pensa a un team. C'è un **Supervisor** (un LLM-router) che analizza il compito e lo delega all'agente giusto: l'**Analista Dati** per le query SQL, il **Ricercatore Web** per le informazioni online, ecc. Ogni agente è un grafo a sé stante, e il Supervisor orchestra la loro collaborazione.
  * **Plan and Execute:** Utile per compiti lunghi. Un agente **Planner** crea una lista di passaggi. Un agente **Executor** (spesso più semplice) esegue quei passaggi senza dover ri-pianificare a ogni iterazione, rendendo il processo più efficiente e prevedibile.

### Language Agent Tree (LAT): L'Esplorazione Multi-percorso

Questa è un'architettura all'avanguardia per risolvere problemi complessi e aperti che non hanno un percorso di soluzione lineare. Se ReAct è un detective che segue un indizio alla volta, **LAT è una squadra di detective che esplora simultaneamente più piste**.

1.  **Espansione (Expansion):** Di fronte a un problema ("scrivi un saggio sull'impatto dell'IA sul lavoro"), l'agente non sceglie una sola azione, ma ne genera **multiple**: "1. Cerca dati economici sull'automazione", "2. Trova articoli accademici sulla riqualificazione della forza lavoro", "3. Analizza le tendenze dei siti di annunci di lavoro". Ognuna di queste diventa un **ramo** in un albero di possibilità.
2.  **Valutazione (Evaluation):** Un LLM "giudice" valuta ogni ramo, assegnandogli un punteggio di promettezza.
3.  **Esplorazione e Potatura (Pruning):** L'agente esplora i rami con il punteggio più alto. Se un percorso si rivela un vicolo cieco, quel ramo viene "potato" (abbandonato).
4.  **Riflessione (Reflection):** Periodicamente, l'agente si ferma, osserva l'intero albero di informazioni che ha costruito e si pone domande di alto livello: "Quali temi comuni emergono dai dati economici e dagli articoli accademici? C'è una contraddizione? Quale dovrebbe essere la tesi principale del mio saggio ora?". Questa riflessione genera un piano nuovo e più informato.

LAT è computazionalmente intensivo, ma è un passo avanti verso un ragionamento più simile a quello umano, basato sull'esplorazione, la valutazione di alternative e la sintesi strategica, perfetto per compiti che richiedono creatività e una profonda comprensione del problema.
