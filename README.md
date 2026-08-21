# 📚 Semantic Book Recommender

An AI-powered book recommendation system that uses **semantic search, zero-shot text classification, emotion analysis, and vector embeddings** to recommend books based on what a reader is looking for rather than relying only on keywords or genres.

Users can describe the kind of book they want—for example:

> *A story about forgiveness and second chances*

and receive semantically relevant recommendations. Results can also be filtered by **Fiction/Nonfiction** and ranked according to emotional tone such as **Happy, Surprising, Suspenseful, Angry, or Sad**.

The project is built with **Python, Hugging Face Transformers, LangChain, Ollama, ChromaDB, Pandas, and Gradio**.

---

## ✨ Features

- 🔎 **Semantic book search** using natural-language descriptions
- 🧠 Local embeddings with **Ollama `nomic-embed-text`**
- 🗃️ Vector similarity search using **ChromaDB**
- 🏷️ **Zero-shot classification** of books into Fiction and Nonfiction
- 🎭 **Emotion analysis** of book descriptions
- 🎚️ Filter recommendations by category
- 😊 Rank recommendations by emotional tone
- 🖼️ Book cover gallery with title, author, and description
- 📓 Jupyter notebooks covering the complete data-processing and ML pipeline
- ⚡ GPU acceleration through PyTorch/CUDA for Hugging Face models

---

## 🧠 How It Works

The application combines several NLP techniques rather than using a traditional collaborative-filtering recommender.

```text
7K Books Dataset
       │
       ▼
Data Cleaning & Exploration
       │
       ▼
Fiction / Nonfiction Classification
       │
       ▼
Emotion Classification
       │
       ▼
Book Description Embeddings
       │
       ▼
Chroma Vector Store
       │
       ▼
Semantic Search
       │
       ├── Category Filter
       └── Emotional Tone Ranking
                 │
                 ▼
             Gradio UI
```

### 1. Data preparation

The original book dataset is downloaded from Kaggle and explored using Pandas.

Records with unsuitable or incomplete descriptions are cleaned, and only descriptions containing sufficient text are retained for subsequent NLP processing.

The cleaned dataset contains metadata including:

- ISBN
- title
- author
- category
- description
- publication year
- average rating
- number of pages
- ratings count
- thumbnail URL

A `tagged_description` field combines the ISBN with the description so vector-search results can later be mapped back to the corresponding book metadata.

---

### 2. Zero-shot book classification

The original dataset contains many inconsistent and highly specific category labels.

To provide a simpler user-facing filter, the project classifies books into:

```text
Fiction
Nonfiction
```

using the Hugging Face model:

```text
facebook/bart-large-mnli
```

through the Transformers zero-shot classification pipeline.

The classifier is validated against books whose categories are already known and achieves approximately:

```text
Accuracy: 77.83%
```

The model is then used to infer categories for books where the simplified category cannot be determined directly.

---

### 3. Emotion analysis

Book descriptions are analysed with:

```text
j-hartmann/emotion-english-distilroberta-base
```

The model produces emotion scores which are added to the book dataset.

The recommendation UI maps these emotions to reader-friendly tones:

| UI Tone | Emotion Score |
|---|---|
| Happy | Joy |
| Surprising | Surprise |
| Angry | Anger |
| Suspenseful | Fear |
| Sad | Sadness |

This allows semantic recommendations to be reordered according to the emotional experience the reader wants.

---

### 4. Semantic embeddings

Each cleaned book description is converted into a vector embedding using:

```text
nomic-embed-text
```

running locally through **Ollama**.

Embeddings are generated in batches and stored in a **Chroma** vector database.

```python
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
```

The project uses batches of 100 documents when populating the vector store.

---

### 5. Semantic recommendation

When a user enters:

```text
A story about forgiveness
```

the query is embedded using the same embedding model.

Chroma then performs similarity search against the book-description vectors:

```python
db_books.similarity_search(query)
```

The most semantically similar books are retrieved.

The system can subsequently:

1. filter by Fiction or Nonfiction;
2. sort by the requested emotional tone;
3. return the highest-ranked books;
4. display their covers and descriptions in the Gradio interface.

---

## 🖥️ Application

The Gradio dashboard allows the user to provide three inputs:

### Book description

Describe the type of book you want:

```text
A story about forgiveness
```

### Category

Choose:

```text
All
Fiction
Nonfiction
```

### Emotional tone

Choose:

```text
All
Happy
Surprising
Angry
Suspenseful
Sad
```

The application displays the recommendations as a gallery containing book covers along with the title, author, and a shortened description.

---

## 🗂️ Repository Structure

```text
book-recommender/
│
├── data-exploration.ipynb
│   └── Dataset download, exploration, cleaning and preprocessing
│
├── text-classification.ipynb
│   └── Fiction/Nonfiction zero-shot classification
│
├── sentiment-analysis.ipynb
│   └── Emotion classification of book descriptions
│
├── vector-search.ipynb
│   └── Embedding generation, Chroma indexing and semantic search
│
├── gradio-dashboard.py
│   └── Gradio recommendation application
│
├── cover-not-found.jpg
│   └── Fallback image for books without covers
│
├── pyproject.toml
│   └── Python project configuration and dependencies
│
├── uv.lock
│   └── Reproducible dependency lock file
│
└── README.md
```

The preprocessing notebooks generate several intermediate files:

```text
books_cleaned.csv
        │
        ▼
books_with_categories.csv
        │
        ▼
books_with_emotions.csv
```

The vector-search workflow additionally creates:

```text
tagged_description.txt
```

These generated files are consumed by later stages of the pipeline.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application and data processing |
| Pandas | Dataset manipulation |
| Jupyter | Data exploration and ML experimentation |
| Hugging Face Transformers | Classification and emotion models |
| PyTorch | Transformer model execution / CUDA support |
| LangChain | Embedding and document abstractions |
| Ollama | Local embedding-model execution |
| `nomic-embed-text` | Semantic embeddings |
| ChromaDB | Vector storage and similarity search |
| Gradio | Interactive frontend |
| KaggleHub | Dataset download |
| uv | Python dependency and environment management |

---

## 📦 Dataset

The project uses the Kaggle dataset:

**7K Books with Metadata** by Dylan J. Castillo

Dataset identifier:

```text
dylanjcastillo/7k-books-with-metadata
```

It is downloaded from the notebook using:

```python
import kagglehub

path = kagglehub.dataset_download(
    "dylanjcastillo/7k-books-with-metadata"
)
```

The raw dataset itself does not need to be committed to the repository.

---

## 🚀 Getting Started

### Prerequisites

You will need:

- Python **3.12 or 3.13**
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- An NVIDIA GPU is recommended for running the Hugging Face models efficiently

The project currently targets:

```text
Python >=3.12,<3.14
```

---

### 1. Clone the repository

```bash
git clone https://github.com/sanbhattdeep/book-recommender.git
cd book-recommender
```

---

### 2. Install dependencies

The project uses `uv` for dependency management:

```bash
uv sync
```

This creates the virtual environment and installs the versions resolved in `uv.lock`.

---

### 3. Install and start Ollama

Install Ollama and verify that it is available:

```bash
ollama --version
```

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Verify:

```bash
ollama list
```

You should see:

```text
nomic-embed-text
```

Make sure Ollama is running before executing the vector-search notebook or starting the Gradio application.

---

### 4. Optional: configure Hugging Face authentication

The Hugging Face models used by this project are public and can be downloaded without authentication.

For higher Hub rate limits, create a Hugging Face access token and add it to a `.env` file:

```env
HF_TOKEN=hf_your_token_here
```

Do not commit `.env` to Git.

---

## 📓 Run the Data Pipeline

If the generated CSV/text files are not already available locally, run the notebooks in this order.

### 1. Data exploration

```text
data-exploration.ipynb
```

Performs data exploration and preprocessing and generates:

```text
books_cleaned.csv
```

### 2. Text classification

```text
text-classification.ipynb
```

Runs Fiction/Nonfiction classification and generates:

```text
books_with_categories.csv
```

### 3. Sentiment analysis

```text
sentiment-analysis.ipynb
```

Adds emotion scores and generates:

```text
books_with_emotions.csv
```

### 4. Vector search

```text
vector-search.ipynb
```

Creates:

```text
tagged_description.txt
```

and demonstrates semantic embedding and Chroma similarity search.

---

## ▶️ Run the Application

Once the generated data files are available and Ollama is running:

```bash
uv run python gradio-dashboard.py
```

Gradio will print the local application URL, typically similar to:

```text
http://127.0.0.1:7860
```

Open it in your browser and enter a description of the kind of book you would like to read.

---

## 🔍 Example Queries

Try queries such as:

```text
A story about forgiveness and redemption
```

```text
A suspenseful story involving crime and investigation
```

```text
A book about personal growth and finding purpose
```

```text
A fantasy adventure involving magic and dangerous journeys
```

```text
A historical book about war and political conflict
```

Unlike keyword matching, semantic search attempts to retrieve books whose descriptions are conceptually similar to the query.

---

## 🎯 What This Project Demonstrates

This project is primarily an exploration of building an end-to-end NLP recommendation pipeline.

It demonstrates:

- cleaning real-world textual datasets;
- working with missing and inconsistent labels;
- evaluating zero-shot classification;
- enriching data using pretrained transformer models;
- generating local embeddings;
- building and querying a vector database;
- combining vector similarity with structured filters;
- adding emotion-aware ranking;
- exposing an ML workflow through an interactive UI.

---

## ⚠️ Current Limitations

- The recommender is **content-based**, not personalized from individual reading history.
- Semantic similarity depends heavily on the quality of the book descriptions.
- Fiction/Nonfiction classification is model-generated for records whose category cannot be mapped directly.
- Emotion scores are inferred from descriptions and do not necessarily represent how every reader will experience a book.
- The Chroma vector store is currently created in memory when the application starts, so embeddings are rebuilt when the application is restarted.
- Running the Hugging Face models on CPU will be considerably slower than running them with CUDA.
- The application expects the generated preprocessing artifacts to exist before it is launched.

---

## 🔮 Possible Improvements

Potential next steps include:

- persist the Chroma vector database between application runs;
- add author, rating, year, and page-count filters;
- combine semantic similarity with book ratings;
- store ISBNs directly as Chroma metadata;
- add automated evaluation for recommendation quality;
- cache embedding generation;
- add tests for preprocessing and retrieval;
- package preprocessing as Python modules instead of notebook-only workflows;
- add Docker support;
- deploy the Gradio application;
- add personalized recommendations based on reading history or user feedback.

---

## 📄 License

No license has currently been specified for this repository.

If the project is intended for reuse or contribution, consider adding an open-source license such as MIT.

---

## 🙌 Acknowledgements

This project makes use of:

- the **7K Books with Metadata** Kaggle dataset;
- Hugging Face pretrained transformer models;
- Ollama and `nomic-embed-text`;
- LangChain;
- ChromaDB;
- Gradio.

---

## 👤 Author

**Sandeep Bhatt**

GitHub: [@sanbhattdeep](https://github.com/sanbhattdeep)

---

If you found the project useful, consider giving the repository a ⭐.