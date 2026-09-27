# How to Explain the Technical Documentation RAG Project

## The one-sentence description

“I built a retrieval-augmented generation system that lets users ask questions about private technical documentation and receive concise answers grounded in the most relevant source passages.”

## A strong 90-second interview explanation

“The problem I wanted to solve was that engineers can spend a lot of time searching through long technical manuals, while a normal chatbot may produce convincing but unsupported answers. I therefore built a RAG system that keeps the documentation at the centre of every response.

During ingestion, I extracted text from PDF and text documents, cleaned it, and split it into overlapping chunks. I generated a semantic embedding for every chunk using a Sentence Transformer model and stored the normalised vectors in a FAISS index, together with metadata such as the document name and page number.

When the user asked a question, I embedded the question with the same model and used cosine similarity in FAISS to retrieve the most relevant chunks. I passed only those chunks and the question to the language model with instructions to use the context, cite its sources, and explicitly say when the answer was not present. I exposed the workflow through FastAPI, packaged it with Docker, and used an AWS deployment pattern so it could run as a service.

My main focus was reliability rather than simply generating fluent text: the interface returned the supporting passages, and I separated retrieval quality from generation quality so that errors could be diagnosed systematically.”

## The problem, your role, contributions, and outcome

| Area | What to say |
|---|---|
| Problem | Technical knowledge was distributed across long private documents; keyword search missed semantically related wording, while a general LLM could hallucinate. |
| Your role | Designed and implemented the end-to-end prototype: ingestion, embeddings, vector search, API, grounding controls, containerisation, and deployment structure. |
| Specific contributions | Built PDF/text loading and chunking; generated embeddings; implemented FAISS retrieval; constructed the grounded prompt; returned citations; exposed endpoints in FastAPI; created the Docker image. |
| Outcome | Users could retrieve relevant evidence and obtain a concise, source-grounded response instead of manually searching whole documents. |
| Engineering relevance | The same pattern could help engineers locate procedures, prior investigation notes, troubleshooting guidance, and equipment documentation, subject to access and safety controls. |

## Explain each technology

### Embeddings

An embedding is a numeric representation of meaning. Texts with similar meaning should have vectors that are close even if they do not use exactly the same words. Both document chunks and questions must be embedded by the same model so their vectors are comparable.

### FAISS

FAISS is a library for efficient nearest-neighbour search over vectors. This prototype normalises the vectors and uses `IndexFlatIP`; inner product on normalised vectors is equivalent to cosine similarity. FAISS returns vector positions, which are mapped back to filenames, pages, and source text using stored metadata.

### Chunking and overlap

Whole manuals are too long and too broad to retrieve as one item. Small chunks improve precision, but splitting can break a sentence or procedure. Overlap repeats a small amount of text between neighbouring chunks so important context near a boundary is less likely to be lost.

### FastAPI

FastAPI exposes the pipeline as a service. `/documents/upload` handles ingestion, `/ask` accepts a question and returns the answer plus sources, and `/health` supports monitoring. It also generates an interactive OpenAPI interface automatically.

### Docker

Docker packages the application and its dependencies into the same reproducible environment for development and deployment. It does not by itself provide scaling, authentication, or monitoring; those are deployment concerns.

### AWS

A credible production mapping is S3 for encrypted documents, ECR for the container image, ECS/Fargate or EC2 for compute, Secrets Manager for credentials, CloudWatch for logs, and EFS or a controlled rebuild process for the index. Only state that you used services you actually used.

## Likely follow-up questions

### Why RAG instead of fine-tuning?

RAG is better suited to frequently changing private knowledge because documents can be updated without retraining a model. It also provides traceable evidence. Fine-tuning is more appropriate for changing behaviour or style than for reliably injecting a large, changing document collection.

### How did you reduce hallucinations?

- Restricted the prompt to retrieved context.
- Required citations to numbered passages.
- Told the model to abstain when the context was insufficient.
- Returned the raw retrieved passages for human verification.
- Kept the generation temperature/settings conservative where supported.
- Evaluated retrieval and answer faithfulness separately.

Do not say hallucinations were eliminated. They were reduced and made easier to detect.

### How would you evaluate it?

Create a labelled set of real questions with expected relevant documents and reference answers. Measure retrieval recall@k or MRR, then evaluate answer correctness, faithfulness to the retrieved context, citation accuracy, abstention quality, latency, and user usefulness. Compare different chunk sizes, overlap values, embedding models, and `top_k` settings.

### Why FAISS rather than a traditional database?

A relational database is suitable for exact filters and structured fields, while FAISS searches by semantic vector similarity. A production system could use both: SQL or metadata filters to restrict the allowed document set, then FAISS for semantic retrieval within it.

### What happens if the correct passage is not retrieved?

The generator cannot reliably recover it. That is why retrieval evaluation matters. Improvements include better chunking, a stronger embedding model, metadata filtering, hybrid keyword-plus-vector search, query rewriting, retrieving more candidates, and reranking them before generation.

### How would you protect private engineering documents?

Use authentication, role-based document permissions, encryption in transit and at rest, private networking, Secrets Manager, audit logs, data-retention controls, and an approved model endpoint. Apply authorisation before retrieval so the model never receives a passage the user is not permitted to see.

### Could the chatbot control a turbine?

No. It is a knowledge-retrieval and decision-support tool. It should not override protection systems, approved operating procedures, or an authorised engineer. High-impact recommendations require source visibility and human review.

### What was the most difficult part?

A strong answer is: “Balancing retrieval precision with sufficient context. Smaller chunks were precise but sometimes incomplete, while larger chunks added noise. I treated chunk size, overlap, and top-k as parameters and inspected retrieval failures rather than assuming the LLM was the only source of error.”

## A slide-ready version

**Problem:** Engineers need fast access to knowledge distributed across private technical documents; ordinary chatbots can return unsupported answers.

**My role:** Designed and implemented the complete RAG prototype and deployment structure.

**Pipeline:** Documents → clean and chunk → embeddings → FAISS → retrieve top-k → grounded LLM prompt → answer with sources.

**Technologies:** Python, Sentence Transformers, FAISS, FastAPI, Docker, AWS, optional OpenAI Responses API.

**Reliability measures:** Local embeddings, source metadata, citations, abstention behaviour, visible retrieved passages, and human verification.

**Relevance to Mitsubishi Power:** The pattern could accelerate access to troubleshooting procedures and prior technical knowledge while maintaining traceability and engineering oversight.

## Integrity check before the interview

Mark each statement as one of:

- **I implemented this:** safe to describe in the past tense and answer in detail.
- **It is in the new prototype:** describe it as a prototype you built to develop the idea further.
- **I would add this in production:** describe it as a proposed improvement, not completed work.

Interviewers usually accept a small prototype. They react badly to a deployment claim that becomes vague under follow-up questions. Clear boundaries make the explanation more credible.

