---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: adapter
language:
- en
pipeline_tag: text-generation
tags:
- lora
- gguf
- llama.cpp
- full-duplex
- voice
- micro-turn
- qwen
---

# Qwen3.6-35B-A3B MicroTurn Duplex LoRA

A small LoRA that teaches a normal 35B text LLM to take part in a live voice call where both sides can talk at the
same time. There is no voice activity detector in the loop: every 0.6 seconds the model reads the few words the
microphone just heard and either writes a few words to say next or writes `<listen>`.

It is the adapter from the Codacus video **"Can a Local LLM Speak and Listen at the Same Time?"**. In that video the
35B runs on one RTX 3060 12GB, and the adapter was trained in about 3 hours on two of them.

The micro-turn idea comes from [DuplexCascade](https://arxiv.org/abs/2603.09180) (Yang, Fujita and Sudo, 2026). This
adapter applies it to a larger model on consumer cards.

## Files

| File | What |
|---|---|
| `microturn-duplex-lora.gguf` | The adapter in llama.cpp's LoRA format (15 MB, F32, rank 32, alpha 64, 3.83M parameters) |
| `microturn_client.py` | A text client with no dependencies: you type what the microphone "heard" per tick and see what the model says |

## What it changes

Only the last four layers of the model (36 to 39: three delta-net layers and one full-attention layer). The adapter
sits on their attention, delta-net and shared-expert projections. Everything below layer 36 is untouched, so the
model keeps what it knows and learns one new thing, which is timing.

A prompt alone gets part of this behaviour out of the base model. Where it fails is the pause in the middle of a
sentence: the base model waited through such a pause 92 times out of 474 on held-out calls. With the first trained
adapter it waited 456 times.

## How to run it

You need a llama.cpp build that supports Qwen3.6-35B-A3B, and the quantisation the adapter was trained against:
[`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF). Other quantisations are
untested.

```bash
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --lora microturn-duplex-lora.gguf \
  -ngl 99 --n-cpu-moe 24 -c 32768 -ub 512 --port 8080
```

`--n-cpu-moe 24` keeps the experts of 24 layers in system RAM so the rest fits on a 12 GB card. On one RTX 3060 this
answers a tick in about 230 ms at the median, well inside the 600 ms tick.

Then:

```bash
python microturn_client.py http://127.0.0.1:8080 --demo
```

Real output of that command with this adapter:

```text
  heard: Hey I'm cleaning       said: <listen>                     P(listen) 0.94
  heard: the bathroom           said: <listen>                     P(listen) 0.97
  heard: right now and          said: <listen>                     P(listen) 0.92
  heard: I got some             said: <listen>                     P(listen) 0.98
  heard: bleach here            said: <listen>                     P(listen) 0.60
  heard: <no voice>             said: <listen>                     P(listen) 0.34
  heard: and I'm going          said: <listen>                     P(listen) 0.86
  heard: to mix it              said: <listen>                     P(listen) 0.93
  heard: with the ammonia       said: <listen>                     P(listen) 0.57
  heard: cleaner so it          said: <listen>                     P(listen) 0.51
  heard: gets extra strong      said: <listen>                     P(listen) 0.37
  heard: <no voice>             said: Do not mix                   P(listen) 0.09
  heard: <no voice>             said: those chemicals.             P(listen) 0.00
  heard: <no voice>             said: It creates toxic             P(listen) 0.30
  heard: <no voice>             said: gas. Please stop.            P(listen) 0.00
  heard: <no voice>             said: <listen>                     P(listen) 0.93
  heard: <no voice>             said: <listen>                     P(listen) 1.00
```

## The protocol

The adapter was trained on one exact text format. Three things have to match.

**1. The system prompt, byte for byte:**

```text
You are a voice assistant in a live phone call; both sides can talk at the same time. Every 0.6 seconds you get what the microphone heard in that moment, as a streaming transcript fragment, or <no voice>. You reply with up to three new words to say next, or <listen> to add nothing.
```

**2. The history, as raw chat-template text sent to `/completion`** (not `/chat/completions`, and with no thinking
block in past assistant turns). Each tick adds one user message and one assistant message, and nothing already sent
is ever edited, so llama.cpp's prompt cache resumes on every request:

```text
<|im_start|>system
{system prompt}<|im_end|>
<|im_start|>user
Hey I'm cleaning<|im_end|>
<|im_start|>assistant
<listen><|im_end|>
<|im_start|>user
<no voice><|im_end|>
<|im_start|>assistant
```

A user message is what speech-to-text produced in the last 0.6 seconds, or `<no voice>` if there was nothing. The
assistant message is the words to speak next, or `<listen>`.

**3. The speak-or-listen rule.** Plain greedy decoding makes the model stop in the middle of its own sentences. The
reason is that `<listen>` is one token, while "keep talking" is spread over many possible next words. In one case the
model was 73% sure it should keep talking and still stopped, because `<` alone held 27% and no single word had more.
So the client reads the probability of `<` on the first token and listens only if it is at least 0.3. Below that it
bans `<` with `logit_bias` and lets the model speak. `microturn_client.py` shows this in about ten lines.

For a full voice call, put streaming speech-to-text in front (the video uses Kyutai `stt-1b-en_fr`) and any
text-to-speech behind it. The model is never told what audio is still playing.

## Training

- **Method.** The lower 36 layers are frozen, so their output was computed once with llama.cpp and cached to disk.
  Only layers 36 to 39 were then run and trained in PyTorch, with their weights unpacked from the same GGUF that is
  served. A check of that reimplementation against llama.cpp's own layer-39 output gave cosine 0.9996.
- **Data.** 564 simulated phone calls (28 more held out), written by local LLMs (Qwen3.8-Flash, and the base 35B for
  the longer calls) and timed into 0.6 s ticks with speech-recognition noise added. There are no recordings of real
  people in it. The calls cover questions, stories told in parts, long explanations with the listener reacting or
  cutting in, pauses in the middle of a thought, and dangerous plans the assistant should interrupt.
- **Loss.** Cross-entropy on the reply tokens with a weight per kind of tick. Most ticks are "the user is talking,
  stay silent", and without weights the model learns to stay silent always.
- **Run.** Rank 32, 282 steps of 4 calls, about 39 seconds per step, AdamW at 2e-4, the first 8,000 tokens of each
  call. About 3 hours on two RTX 3060 12GB.

This is the fifth run. The earlier ones each failed in their own way (one repeated its warning twenty times in a
row, one said "mm-hm" 950 times where a good listener would have said it 164 times), and the video walks through
what was wrong with the data each time.

## Results

Closed-loop replay of held-out calls: the model sees its own earlier replies, as it does in a real call. Decoding as
described above (temperature 0, listen threshold 0.3).

| | Long calls (16) | Short calls (12) |
|---|---|---|
| Starts its answer within one tick of the right moment | 121 / 156 (78%) | 137 / 221 (62%) |
| Interrupts a dangerous plan | 6 / 10 | 6 / 17 |
| Stops talking when the user cuts in | 7 / 7 | 27 / 30 |
| Fillers ("mm-hm") placed where a good listener would | 123 / 351 (35%) | 54 / 164 (33%) |
| Talks on a tick where the user is talking or pausing mid-thought | 8% of those ticks | 5% of those ticks |
| Stops in the middle of its own sentence | 1% of its utterances | 0% |

## What it still gets wrong

- **It is slow to start.** In the full voice demo the first audible word arrives about 3 seconds after you stop
  talking. Models built for this, like Moshi, start in about a quarter of a second.
- **It does not always speak up when it should.** It caught 6 of 10 dangerous plans on the longer test calls.
- **Its fillers land in the right spot about a third of the time.**
- English only. It was trained on text that imitates one speech-to-text model's fragments, so a recogniser with very
  different chunking may behave differently.
- It is a research demo trained on a few hundred written conversations. Do not rely on it to warn anyone about
  anything.

## Licence

Apache 2.0, the same as the base model.
