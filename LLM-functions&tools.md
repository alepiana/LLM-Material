### Summary

The landscape of LLMs and the libraries that support them has evolved rapidly in recent months. This course is designed to keep you ahead of these changes. 

You’ll explore new advancements like ChatGPT’s function calling capability, and build a conversational agent using a new syntax called LangChain Expression Language (LCEL) for tasks like tagging, extraction, tool selection, and routing.

After taking this course, you’ll know how to: 

Generate structured output, including function calls, using LLMs;

Use LCEL, which simplifies the customization of chains and agents, to build applications;

Apply function calling to tasks like tagging and data extraction;

Understand tool selection and routing using LangChain tools and LLM function calling – and much more.

Start applying these new capabilities to build and improve your applications today.
-----

### Lo Scenario: Perché Questo Cambiamento Era Necessario?

Prima di addentrarci nei dettagli tecnici, capiamo il "perché". Nei primi tempi di LangChain, l'approccio per costruire catene (`chains`) e agenti (`agents`) era principalmente **imperativo**. Si istanziavano classi come `LLMChain`, si passavano i componenti (modello, prompt) come argomenti del costruttore e poi si eseguiva il tutto con un metodo `.run()` o `.predict()`.

Questo funzionava bene per compiti semplici, ma diventava macchinoso e poco flessibile quando si volevano creare flussi complessi, personalizzare il comportamento o preparare l'applicazione per un ambiente di produzione (con esigenze di streaming, logging, parallelizzazione, etc.).

I due progressi menzionati nel testo, **Function Calling** e **LangChain Expression Language (LCEL)**, sono la risposta a queste limitazioni. Rappresentano un cambio di paradigma verso un approccio **dichiarativo**, più intuitivo e infinitamente più componibile.

-----

## 1\. ChatGPT’s Function Calling: L'LLM Che Chiede Gli Strumenti Giusti 🛠️

Questo è il progresso fondamentale a livello del modello stesso che ha sbloccato tutto il resto.

#### Spiegazione Semplice

Immagina un LLM come un consulente estremamente intelligente ma chiuso in una stanza senza mani. Può solo parlare. Se gli chiedi "Che tempo fa a Milano?", lui può *solo* rispondere "Per saperlo, dovrei controllare un servizio meteo". A quel punto, tu (sviluppatore) devi "sentire" questa risposta, interpretarla, chiamare l'API del meteo e poi riportare il risultato al consulente.

Il **Function Calling** (ora chiamato più genericamente **Tool Calling** da OpenAI) dà al consulente un "catalogo di bottoni" che può premere. Invece di rispondere a parole, l'LLM può dire: "Ho bisogno che tu prema il bottone `get_current_weather` con l'argomento `location` impostato su `'Milano'`". Questa non è una frase in linguaggio naturale, ma un output strutturato e garantito in formato **JSON**. Questo elimina ogni ambiguità e rende l'integrazione tra LLM e strumenti esterni (API, database, funzioni) incredibilmente robusta.

#### Spiegazione Dettagliata

Tecnicamente, quando interagisci con un modello che supporta il tool calling (come quelli di OpenAI), non gli passi solo il tuo prompt, ma anche uno **schema** JSON che descrive le "funzioni" o "strumenti" che metti a sua disposizione.

Ad esempio, se hai una funzione Python:

```python
def get_current_weather(location: str, unit: str = "celsius"):
    # ... logica per chiamare un'API meteo ...
    return f"La temperatura a {location} è di 25° {unit}."
```

Fornirai al modello una descrizione simile a questa:

```json
{
  "name": "get_current_weather",
  "description": "Ottieni la temperatura attuale in una data località",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "La città, es. Milano"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"]
  }
}
```

Quando l'utente chiede "Che tempo fa a Milano?", il modello non risponde a parole, ma restituisce un oggetto JSON che ti dice di chiamare la funzione `get_current_weather` con gli argomenti corretti.

In LangChain, non devi scrivere questi JSON a mano. Puoi legare gli strumenti al modello in modo molto più semplice, usando comandi come `.bind_tools()`:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_current_weather(location: str, unit: str = "celsius"):
    """Ottieni la temperatura attuale in una data località."""
    # ... logica API ...
    return f"La temperatura a {location} è di 25° {unit}."

# Associamo lo strumento al modello
model_with_tools = ChatOpenAI(model="gpt-4o").bind_tools([get_current_weather])
```

Ora `model_with_tools` è un modello "potenziato" che sa quando e come chiedere di usare lo strumento `get_current_weather`.

-----

## 2\. LangChain Expression Language (LCEL): La "Pipe" per Comporre Flussi ⛓️

LCEL è la nuova sintassi che sfrutta appieno capacità come il Tool Calling. È il "collante" che ti permette di costruire catene in modo dichiarativo.

#### Spiegazione Semplice

L'idea centrale di LCEL è l'operatore "pipe" (`|`), preso in prestito dalle shell Unix. Significa "prendi l'output del componente a sinistra e passalo come input al componente a destra".

Invece di scrivere:
`chain = LLMChain(llm=model, prompt=prompt_template, output_parser=parser)`

Con LCEL scrivi in modo molto più naturale:
`chain = prompt_template | model | parser`

Questa non è solo una semplificazione sintattica. Una catena costruita con LCEL ottiene automaticamente una serie di funzionalità pronte per la produzione:

  * **Streaming asincrono:** Puoi ottenere i risultati un token alla volta.
  * **Esecuzione parallela:** LangChain può eseguire in parallelo i componenti che non dipendono l'uno dall'altro.
  * **Tracciabilità e Debug:** Si integra perfettamente con **LangSmith**, permettendoti di visualizzare ogni passo della catena.
  * **Componibilità estrema:** Ogni pezzo di una catena LCEL è a sua volta una "catena" (un oggetto `Runnable`), quindi puoi combinare e innestare catene complesse con facilità.

-----

## 3\. Applicazioni Pratiche: LCEL e Tool Calling in Azione

Vediamo come questi due concetti lavorano insieme per realizzare i compiti menzionati nel testo.

#### Generare Output Strutturato (Tagging & Estrazione)

Questo è uno degli usi più potenti. Vogliamo che l'LLM non restituisca testo libero, ma un output JSON strutturato secondo un nostro schema.

**Obiettivo:** Estrarre nome, età e interessi di una persona da un testo.

**Come funziona:** Definiamo una classe (usando la libreria **Pydantic**) che rappresenta la nostra struttura dati. LangChain usa questa classe per creare dinamicamente uno schema di "tool" e lo passa al modello, istruendolo a "chiamare questo tool" con i dati estratti.

**Comandi e Codice (LCEL):**

```python
from langchain_core.pydantic_v1 import BaseModel, Field

# 1. Definisci la struttura dati desiderata
class UserInfo(BaseModel):
    name: str = Field(description="Nome della persona")
    age: int = Field(description="Età della persona")
    interests: list[str] = Field(description="Una lista dei suoi interessi")

# 2. "Lega" la struttura al modello
structured_llm = ChatOpenAI(model="gpt-4o").with_structured_output(UserInfo)

# 3. Costruisci la catena con LCEL
prompt = ChatPromptTemplate.from_messages([
    ("system", "Estrai le informazioni dell'utente dal testo seguente."),
    ("human", "{input_text}")
])

extraction_chain = prompt | structured_llm
```

Quando esegui `extraction_chain.invoke({"input_text": "Mi chiamo Marco, ho 30 anni e mi piacciono il calcio e i videogiochi."})`, l'output non sarà una stringa, ma un'istanza della classe `UserInfo` con i campi correttamente popolati. Il comando chiave qui è `.with_structured_output()`, che è una scorciatoia LCEL per implementare l'estrazione tramite tool calling.

#### Tool Selection e Routing

Questo è il modo moderno di costruire agenti e `Router Chains`.

**Obiettivo:** Creare un assistente che possa sia cercare sul web sia rispondere a domande generiche.

**Come funziona:** Definiamo diversi strumenti (es. `search_web`, `get_weather`). Li leghiamo al modello usando `.bind_tools()`. La catena LCEL passerà l'input dell'utente al modello, che deciderà quale strumento (se presente) è più appropriato chiamare. La logica successiva eseguirà lo strumento scelto e, se necessario, passerà il risultato di nuovo all'LLM per una risposta finale.

**Logica e Comandi:**

```python
# Abbiamo già definito 'get_current_weather' come @tool
# Definiamone un altro
@tool
def search_web(query: str):
    """Cerca informazioni sul web."""
    # ... logica per usare un'API di ricerca ...
    return f"Risultati per '{query}': ..."

# 1. Lega entrambi gli strumenti al modello
tools = [get_current_weather, search_web]
model_with_tools = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# La logica di routing è intrinseca nel modello
# Se l'input è "Che tempo fa a Roma?", model_with_tools restituirà una ToolCall a get_current_weather
# Se l'input è "Chi ha vinto l'ultimo mondiale?", restituirà una ToolCall a search_web
```

Il **routing** qui non è più gestito da una `RouterChain` esplicita, ma è delegato all'intelligenza del modello stesso, che sceglie lo strumento giusto grazie alla capacità di Tool Calling. Questo rende il sistema più flessibile e meno rigido rispetto all'approccio precedente.

In sintesi, il testo descrive il passaggio a un'architettura più potente e flessibile. **LCEL (`|`)** è la nuova sintassi per costruire, mentre il **Tool Calling** (gestito da comandi come `.bind_tools()` e `.with_structured_output()`) è la capacità del modello sottostante che permette all'LLM di interagire con il mondo esterno e produrre dati strutturati in modo affidabile.

Certamente. Aggiungo una sezione conclusiva dedicata agli **Agenti Conversazionali**, che rappresentano la sintesi e l'applicazione più avanzata di tutti i concetti che abbiamo discusso finora, come il Tool Calling e LCEL.

-----

### 4\. Agenti Conversazionali: L'Orchestrazione Intelligente di Ragionamento e Azione 🤖

Se le catene (chains) sono come delle ricette che l'LLM segue pedissequamente, gli **agenti** sono come degli chef a cui dai un obiettivo e la libertà di scegliere quali ingredienti e utensili usare per raggiungerlo. Un agente è un sistema che utilizza un LLM non solo per generare testo, ma come un vero e proprio **motore di ragionamento** per decidere autonomamente una sequenza di azioni da compiere.

#### Le Basi dell'Agente: LLM + Codice

Fondamentalmente, un agente è l'unione di due componenti:

1.  **LLM (Il Cervello):** Il modello linguistico funge da cervello decisionale. Data una richiesta dell'utente e un set di strumenti a disposizione, il suo compito è quello di "pensare" e decidere quale sia il passo successivo.
2.  **Codice (Le Mani):** È l'infrastruttura che permette all'agente di *agire* nel mondo. Questo codice si occupa di eseguire gli strumenti che l'LLM ha deciso di usare (es. chiamare un'API, interrogare un database, eseguire un calcolo) e di riportare i risultati all'LLM.

L'LLM non esegue mai direttamente il codice. Si limita a generare un output strutturato (grazie al **Tool Calling**) che indica quale funzione (strumento) il codice deve eseguire e con quali argomenti.

#### Il Ciclo dell'Agente (Agent Loop)

Il cuore pulsante di un agente è il suo ciclo operativo, spesso basato su un framework di pensiero chiamato **ReAct (Reason + Act)**. Questo ciclo si ripete finché l'agente non ritiene di aver risolto il compito.

1.  **Ragionamento e Scelta dello Strumento (Reason & Tool Choice):** L'agente analizza la richiesta dell'utente e la cronologia delle azioni precedenti. Il suo primo passo è il "pensiero" (`Thought`), in cui l'LLM ragiona ad alta voce (per sé stesso) su quale sia il modo migliore di procedere. Basandosi su questo pensiero, decide quale strumento (`Tool`) è il più adatto da usare e con quali parametri (`Tool Input`).
2.  **Azione (Action):** L'infrastruttura di codice esegue lo strumento scelto dall'LLM. Ad esempio, se l'LLM ha deciso di chiamare `search_web(query="ultime notizie AI")`, il codice eseguirà la ricerca.
3.  **Osservazione (Observation):** Il risultato dell'azione (l'output dello strumento) viene catturato. Questo risultato è l'**osservazione**. Ad esempio, l'osservazione potrebbe essere una lista di titoli di notizie.
4.  **Ripetizione:** L'osservazione viene restituita all'LLM, che la usa per iniziare un nuovo ciclo di ragionamento. "Ok, ho questi titoli. Ora devo analizzare il primo per trovare la risposta". Il ciclo `Thought -> Action -> Observation` si ripete, con l'agente che compie passi incrementali verso la soluzione.
5.  **Condizione di Arresto:** Il ciclo termina quando l'LLM, nel suo momento di "pensiero", conclude di avere abbastanza informazioni per formulare la risposta finale (`Final Answer`) all'utente.

#### Il Ruolo Chiave dell' `agent_scratchpad`

Come fa l'agente a ricordare i passi che ha già compiuto? Se al punto 4 del ciclo l'LLM ricevesse solo l'ultima osservazione, non avrebbe contesto sul suo precedente ragionamento.

Qui entra in gioco l'**`agent_scratchpad`** (blocco note dell'agente). È una variabile speciale, una sorta di **memoria a breve termine**, che viene passata al prompt dell'agente a ogni iterazione. L'`agent_scratchpad` accumula la cronologia completa del ciclo `Thought -> Action -> Observation`.

Ad ogni nuovo passo, lo scratchpad potrebbe contenere qualcosa del genere:

```
Thought: Devo cercare le ultime notizie sull'IA per rispondere alla domanda dell'utente.
Action: search_web(query="ultime notizie AI")
Observation: [Titolo 1, Titolo 2, Titolo 3]

Thought: I titoli sono troppo generici. Devo analizzare il contenuto del primo risultato per trovare dettagli specifici.
Action: read_url(url="http://esempio.com/titolo1")
Observation: "Un nuovo modello di IA chiamato 'Phoenix' è stato rilasciato oggi..."
```

Passando questo "blocco note" all'LLM, gli si fornisce tutto il contesto necessario per decidere il passo successivo in modo coerente, evitando di ripetere azioni o di perdere il filo del ragionamento. In LangChain, l'`agent_scratchpad` è una variabile gestita automaticamente dall'**AgentExecutor**, il motore che orchestra l'intero ciclo.
