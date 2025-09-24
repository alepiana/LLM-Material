#### Prompt Engineering - Course Outline

In ChatGPT Prompt Engineering for Developers, you will learn how to use a large language model (LLM) to quickly build new and powerful applications. 
Using the OpenAI API, you’ll be able to quickly build capabilities that learn to innovate and create value in ways that were cost-prohibitive, highly technical, or simply impossible before now. 
This short course taught by Isa Fulford (OpenAI) and Andrew Ng (DeepLearning.AI) will describe how LLMs work, provide best practices for prompt engineering, 
and show how LLM APIs can be used in applications for a variety of tasks, including: 
  - Summarizing (e.g., summarizing user reviews for brevity) 
  - Inferring (e.g., sentiment classification, topic extraction) 
  - Transforming text (e.g., translation, spelling & grammar correction) 
  - Expanding (e.g., automatically writing emails) 
In addition, you’ll learn two key principles for writing effective prompts, how to systematically engineer good prompts, and also learn to build a custom chatbot.
All concepts are illustrated with numerous examples, which you can play with directly in our Jupyter notebook environment to get hands-on experience with prompt engineering

---

# Lezione: Prompt Engineering per LLM – Teoria e Applicazioni Pratiche

Il Prompt Engineering è una disciplina fondamentale per sfruttare al massimo le capacità dei **Large Language Models (LLM)**, come GPT. Essa consiste nel progettare con cura le istruzioni che vengono date al modello per ottenere risposte precise, utili e coerenti. A differenza dei classici modelli di machine learning, con un LLM non è necessario addestrare un modello diverso per ogni compito, ma è possibile guidare un unico modello pre-addestrato a eseguire una vasta gamma di compiti, semplicemente definendo prompt efficaci.

---

## 1. Principi Fondamentali del Prompt Engineering

Alla base del Prompt Engineering ci sono due principi chiave:

### 1.1 Dare istruzioni chiare e specifiche all’agente

Spesso si tende a pensare che “breve” equivalga a “chiaro”, ma non è sempre vero: un prompt può essere breve ma ambiguo. È fondamentale fornire **istruzioni dettagliate e precise**, indicando esattamente cosa ci si aspetta dal modello.

Alcuni accorgimenti pratici per rendere un prompt chiaro e specifico includono:

* **Usare delimitatori**: simboli come `"""..."""`, `...`, `---`, `<...>`, `<tag>...</tag>` aiutano a separare istruzioni e dati, riducendo il rischio di **prompt injection**, ossia istruzioni indesiderate che possono influenzare la risposta.
* **Richiedere output strutturati**: definire un formato chiaro per la risposta, ad esempio liste numerate, tabelle o JSON, aiuta il modello a organizzare le informazioni in modo coerente.
* **Verificare condizioni predefinite**: se il risultato deve rispettare certe condizioni (lunghezza, formato, precisione), specificarle nel prompt.
* **Few-shot prompting**: mostrare esempi di come completare un compito prima di chiedere al modello di eseguirlo autonomamente. Questo aiuta a ridurre ambiguità e aumentare la precisione.

### 1.2 Dare tempo al modello per “pensare”

Un errore comune è aspettarsi che il modello produca la risposta corretta immediatamente. È invece utile **guidare il modello a elaborare passo per passo la soluzione**:

* Specificare i passaggi necessari per completare un compito, ad esempio “prima analizza il testo, poi estrai i punti principali, infine genera il riassunto”.
* Instruire il modello a sviluppare la propria soluzione prima di arrivare a una conclusione finale, simile al ragionamento umano passo-passo.

Questi accorgimenti aumentano significativamente la qualità delle risposte, soprattutto in compiti complessi.

---

## 2. Limitazioni dei Modelli e Hallucination

Nonostante la potenza dei LLM, esistono **limitazioni intrinseche**, tra cui la più significativa è la **hallucination**: il modello può produrre affermazioni plausibili ma false.

**Strategie per ridurre le hallucination**:

* Chiedere al modello di **cercare informazioni rilevanti prima di rispondere**, e poi formulare la risposta basandosi solo su queste informazioni.
* Specificare nel prompt di **verificare la coerenza e la plausibilità** delle affermazioni generate.

---

## 3. Sviluppo Iterativo del Prompt (Iterative Prompt Development)

Il Prompt Engineering è un processo iterativo, simile allo sviluppo di un software:

1. **Idea iniziale**: definire cosa si vuole ottenere.
2. **Implementazione del prompt**: scrivere il testo del prompt e, se necessario, codice o dati.
3. **Risultato sperimentale**: osservare le risposte del modello.
4. **Analisi degli errori**: comprendere perché il risultato non è soddisfacente.
5. **Refinement**: migliorare il prompt e ripetere il ciclo.

**Linee guida** per iterare con successo:

* Essere sempre chiari e specifici.
* Analizzare con attenzione perché il modello non ha prodotto il risultato desiderato.
* Raffinare sia l’idea che il prompt, mantenendo un approccio sistematico.
* Ripetere fino a ottenere risultati coerenti e affidabili.

---

## 4. Altri Accorgimenti nei Prompt

Alcuni dettagli tecnici possono influenzare la risposta:

* Limitazioni su **parole, caratteri, token o frasi**: possono indirizzare il modello verso output più sintetici o dettagliati.
* Focalizzazione: specificare chiaramente su quali elementi il modello deve concentrarsi, evitando risposte che deviano dal tema principale.

---

## 5. Tipologie di Task con LLM

I LLM sono estremamente versatili e possono essere utilizzati per diversi tipi di compiti. Di seguito i principali, con esempi pratici:

### 5.1 Summarizing (Sintesi)

* Obiettivo: ridurre un testo mantenendo i punti principali.
* Prompt tipico: “Riassumi il seguente testo in massimo 100 parole, concentrandoti sul tema X”.
* Nota: a volte un riassunto generico può includere informazioni non rilevanti; in questi casi, **estrarre** le informazioni specifiche può essere più efficace del semplice riassunto.

### 5.2 Inferring (Inferenza)

* Input: testo.
* Output: analisi, classificazioni o estrazione di etichette (es. sentiment, emozioni, topic).
* Vantaggio rispetto al ML tradizionale:

  * Non è necessario addestrare un modello per ogni tipo di output.
  * Serve meno dati e l’addestramento è più semplice.
* Esempi:

  * **Sentiment analysis**: determinare se una recensione è positiva o negativa.
  * **Topic detection**: identificare l’argomento principale di un testo (anche in modalità zero-shot, senza dati etichettati).

### 5.3 Transforming (Trasformazione)

* Obiettivo: modificare o convertire un testo.
* Esempi:

  * Traduzione linguistica.
  * Correzione ortografica e grammaticale.
  * Adjustamento del tono o del formato del testo.
* Prompt efficace: “Riformula il seguente testo in tono formale e corretto grammaticalmente”.

### 5.4 Expanding (Espansione)

* Obiettivo: generare contenuti basati su input specifici.
* Esempi:

  * Generazione automatica di email personalizzate in customer service.
  * Creazione di testi estesi da brevi note.
* Prompt efficace: fornire informazioni contestuali (tono, contenuto, valutazione del cliente) e chiedere al modello di creare la risposta personalizzata.

---

## 6. Parametri e Ruoli nei Messaggi

Nei modelli di dialogo come GPT, è utile definire i **ruoli dei messaggi**:

* **System**: definisce il contesto o le regole generali.
* **User**: input diretto dell’utente.
* **Assistant**: risposte generate dal modello.

Alcuni modelli permettono di settare parametri come la **temperature**, che controlla il grado di creatività o casualità nella risposta: valori più bassi → risposte più conservative e precise; valori più alti → risposte più varie e creative.

---

## 7. Conclusione

Il Prompt Engineering non è solo scrivere una frase breve e generica: è un processo **strategico, iterativo e preciso**. Per ottenere il massimo dai LLM:

* Fornire **istruzioni chiare e specifiche**.
* Dare tempo al modello per “pensare” e sviluppare la risposta passo-passo.
* Usare esempi, delimitatori e output strutturati.
* Iterare continuamente il prompt, analizzando errori e raffinando le istruzioni.
* Essere consapevoli dei limiti del modello, in particolare delle hallucination, e adottare strategie per minimizzarle.

I LLM permettono di gestire compiti complessi di sintesi, inferenza, trasformazione ed espansione senza dover addestrare modelli separati per ogni task, rendendo l’AI più accessibile, flessibile e potente.

---

Se vuoi, posso anche creare **una versione visuale della lezione**, con diagrammi che rappresentano il flusso dell’Iterative Prompt Development, i ruoli dei messaggi, e i diversi tipi di task. Questo rende molto più immediata la comprensione.

Vuoi che lo faccia?
