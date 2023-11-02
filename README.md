# TopicModeling
Progetto di Short Text Topic Modeling.

# Descrizione
Con questo progetto è possibile generare le etichette dei principali argomenti trattati all'interno di una serie di testi brevi.

## Input
I testi in input sono testi brevi, strutturati in una lista di json:
```
[
    {
        "id": 0,
        "text": "testo"
    },
    {
        "id": 1,
        "text": "testo"
    },
    ...
]
```

Per ogni insieme di testi in input, è presente una domanda associata.

## Metodologia
Gli step fondamentali del progetto sono:
- Preprocessing del testo
- Clusterizzazione delle frasi in input
- Definizione del numero di cluster
- Definizione delle etichette

Il primo step prevede la pulizia di alcuni testi non utili per l'obiettivo. In particolare, vengono escluse:
- le frasi in una lingua differente rispetto a quella fornita in input;
- le frasi più corte di 3 caratteri;
- le frasi conteneti solo caratteri speciali.

Il secondo step si basa principalmente sull'utilizzo del  modello [BERTopic](https://maartengr.github.io/BERTopic/index.html). Viene utilizzato il modello multilingual, per poter interagire con diverse lingue. 

Per la selezione del numero di cluster, possono essere utilizzati tre metodi:
- WCSS: Within-Cluster Sum of Squares, rappresenta la somma delle distanze quadrate tra ogni punto di dati all'interno di un cluster e il centroide del cluster a cui appartiene. In altre parole, la WCSS misura quanto ciascun punto di dati all'interno di un cluster è vicino al suo centroide;
- Silhouette: questo indice considera quanto ogni punto di dati in un cluster è simile agli altri punti nello stesso cluster (coesione) rispetto a quanto è dissimile rispetto ai punti nei cluster vicini (separazione);
- Davies-Bouldin:  l'indice di Davies-Bouldin è basato sulla concettualizzazione che cluster di alta qualità dovrebbero avere una bassa dissimilarità tra i punti all'interno dello stesso cluster e una grande dissimilarità tra i punti dei diversi cluster. L'obiettivo è trovare un numero di cluster che minimizzi l'indice.

Il numero massimo di cluster da utilizzare sarà il valore indicato da BERTopic; il numero minimo 3 (ossia 2 cluster + cluster -1). Per ridurre il numero di iterazioni, viene utilizzata una procedura di early stopping in base al metodo di selezione del numero di cluster scelto: se dopo n incrementi nel numero di cluster non cambia il migliore possibile, la procedura viene fermata. N viene definito in input e di default è 6.

Infine, per la definizione delle etichette è stata utilizzata la libreria [Langchain](https://python.langchain.com/docs/get_started/introduction) insieme al modello [gpt-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5). In particolare, sono stati selezionati per ogni cluster i 100 testi più rappresentativi, definiti come i testi più vicini al centroide del cluster, al fine di definire un prompt per interrogare il modello su quale fosse l'etichetta più rappresentativa. Nel prompt è stata inserita inoltre la domanda associata alla serie di risposte. 

E' inoltre possibile visualizzare graficamente i cluster ottenuti.

# Setup

Prima di iniziare, è necessario generare un file .env che conterrà le principali variabili d'ambiente utili per eseguire il progetto. Le variabili disponibili sono:
- FILENAME: obbligatoria, contiene la posizione ed il nome del file in input da utilizzare;
- OPENAI_KEY: obbligatoria, contiene l'api-key di openai che deve essere utilizzata;
- FILENAME_QUESTION: contiene la posizione ed il nome del file che contiene la domanda associata all'input da utilizzare;
- N_CLUST: numero di cluster da computare. Può essere un intero (numero preciso di cluster da utilizzare), "auto" (utilizza il numero di cluster indicati da BERTopic), "compute" (utilizza un metodo di selezione del numero di cluster). Default: auto;
- N_CLUST_METHOD: se N_CLUST è "compute", indicare qui quale dei tre metodi disponibili utilizzare per ridurre il numero di cluster. I valori disponibili sono "wcss", "silhouette", "davies_boulding". Default: wcss;
- EARLY_STOPPING: se N_CLUST è "compute", indicare qui dopo quanti step fermarsi se non cambia il miglior numero di cluster. Default: 6;
- LANGUAGE: lingua del testo in input. Default: italian;
- PLOT_RESULTS: valore True/False per indicare se visualizzare un plot visivo del risultato ottenuto. Default: False.

Un esempio di file .env può essere:
```
OPENAI_KEY=sk-...
FILENAME=data/datasets/json/4.json
FILENAME_QUESTION=data/datasets/sq/4q.txt	 
N_CLUST=compute
N_CLUST_METHOD=wcss	
```

# Installazione

## Tramite docker
E' possibile eseguire il progetto generando un container docker.

Come prima cosa, generare l'immagine:

```docker build -t topic_extract . ```

Infine eseguire il container:

```docker run -d --name topic_container --env-file .env -v [output_directory]:/usr/src/TopicExtractor/output topic_extract```

E' possibile visualizzare i log con il comando:

```docker logs -f topic_container```

## Tramite conda
Come prima cosa, generare un environment dedicato. E' consigliato utilizzare una versione di python >=3.10

```conda create --name nome_ambiente python=3.10```

Attivare l'ambiente appena creato
 
``` conda activate nome_ambiente ```

Spostarsi quindi nella cartella di progetto e installare poi i requirements presenti nel file requirements.txt

``` pip install -r requirements.txt ```

Eseguire infine il file main.py.

# Output
Nella cartella output verranno generati per ogni input due file:
- output_[nome_file].xlsx: questo file conterrà le seguenti colonne
    - text: testo di riferimento
    - topic_id: cluster di apparenza
    - probs: probabilità di appartenenza al cluster
    - embedding: embedding associato al testo
- output_[nome_file]_labels.xlsx: questo file conterrà il mapping tra il topic_id e la descrizione del cluster
