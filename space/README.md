---
title: QUERION
sdk: docker
app_port: 7860
pinned: false
---

# QUERION

A 100M-parameter text-to-SQL model built from scratch, served by its own inference engine
(continuous batching, paged KV cache, prefix cache). Code: https://github.com/AmanSingh-404/QUERION

`POST /generate` takes `{"tables": ["CREATE TABLE ..."], "question": "...", "samples": 4}` and
streams constrained candidate queries as server-sent events. The server never sees table
contents: the browser runs the candidates and votes.