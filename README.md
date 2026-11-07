# Cited Research Agent

A small agent that chooses when to search, read, and answer questions about a local document collection. The model selects actions; Python controls which tools run and checks the final citations.

I built this to explore a common retrieval problem: a plausible answer is not enough if the reader cannot check where it came from.

## Run the demo and tests

Requires Python 3.10+; no Python dependencies.

```bash
python app.py
python -m unittest -v
```

The default demo uses **scripted decisions, not a live LLM**. `example-output.json` is recorded from that demo. Nine local tests pass, covering citation validation, rejected tool calls, loop limits, and the mocked provider contract.

## Use a local model

Install and start Ollama separately, and pull a model that can follow JSON instructions. Create a JSON object mapping document IDs to text; an example is included.

```bash
python app.py --model YOUR_INSTALLED_MODEL --documents documents.example.json --question "How long are audit records kept?"
```

The adapter uses Ollama's documented `/api/chat` endpoint, `format: json`, and non-streaming responses. See [Ollama API](https://docs.ollama.com/api/chat). Live-model execution has **not** been validated in this environment. Model quality depends on the model you run.

## Design

```text
Question -> model decision -> validated action
                            -> search (lexical overlap)
                            -> read (known document ID)
                            -> answer + exact quotes
                            -> abstain
```

- Maximum eight model turns by default; repeated tool calls get validation feedback.
- The model receives document IDs, not filesystem access. No shell or web tools.
- Answers must cite documents already read and include matching source quotes.
- SHA-256 source fingerprints help identify the exact document version.
- Documents are explicitly marked as untrusted evidence.

## What the checks do not prove

An exact quote and valid source ID do not prove that the generated answer follows from the evidence. Prompt injection can still distort answers even though it cannot grant new tools. This project does not provide semantic entailment checks, vector search, authentication, or a production serving layer. Search uses simple term overlap, so paraphrases can be missed. The trace avoids full source text, but model messages contain supplied documents; use only data you intend to send to your local Ollama service.

Network and provider errors fail the CLI run rather than silently fabricating an answer. Add provider retry policies and a broader labeled question set before evaluating operational quality.

Independent portfolio project, added November 2026. Uses demonstration policies, not employer data or production results.
