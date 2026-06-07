# Open WebUI and Ollama RAG Integration

SwarLocal leverages Open WebUI knowledge collections to inject custom RAG (Retrieval-Augmented Generation) context into LLM chats.

## 1. Local RAG Flow
When using the **Local AI (Ollama)** brain:
* If a Knowledge Source is selected in the UI, SwarLocal queries the Open WebUI completions API (`/api/v1/chat/completions`) passing the collection ID.
* Open WebUI retrieves matching chunks, injects them into the prompt context, and returns the response from the local Ollama model.
* If Open WebUI is unreachable or the RAG query fails, SwarLocal will fall back to querying the local Ollama model directly, provided `rag_fallback_to_ollama` is enabled in settings.

## 2. Cloud Provider Direct Chat Flow
* Currently, Open WebUI's context retrieval mechanism is tightly integrated with its local chat endpoint.
* When using a cloud provider (**OpenAI** or **Gemini**), SwarLocal routes queries directly to the cloud completions API to prevent API routing conflicts.
* **Temporary RAG Pathing**: The response payload specifies the `rag_path` used. For local turns, it indicates `local_openwebui`, and for cloud turns, it indicates `cloud_direct_chat`.
* Future releases will introduce a custom retrieval adapter to query Open WebUI documents directly and insert them as text context into OpenAI and Gemini prompt inputs.
