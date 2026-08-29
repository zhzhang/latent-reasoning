Never loop over examples or batches manually as you feed them in to vLLM.
Feed the examples in a single vLLM call and allow vLLM to handle the specific batching efficiently.

Write all code in a way that it can be resumed idempotently, incase the size of the inputs change.