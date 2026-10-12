"""Micro-turn text client for the LoRA: you type what the microphone "heard" in each 0.6 s tick, and the model
replies with a few words to say next, or <listen>.

    python microturn_client.py http://127.0.0.1:8080            # type ticks by hand (empty line = silence)
    python microturn_client.py http://127.0.0.1:8080 --demo     # play a scripted call

The server is llama.cpp's llama-server with the base GGUF and --lora microturn-duplex-lora.gguf. No other packages needed.
The prompt below is the text the adapter was trained on; keep it byte for byte.
"""
import json
import math
import sys
import urllib.request

TICK = 0.6
LISTEN, NOVOICE = "<listen>", "<no voice>"
LISTEN_P = 0.3   # listen only if "<" holds at least this much of the first token's probability
SYSTEM = ("You are a voice assistant in a live phone call; both sides can talk at the same time. Every "
          f"{TICK:g} seconds you get what the microphone heard in that moment, as a streaming transcript fragment, or "
          f"{NOVOICE}. You reply with up to three new words to say next, or {LISTEN} to add nothing.")

DEMO = ["Hey I'm cleaning", "the bathroom", "right now and", "I got some", "bleach here", NOVOICE,
        "and I'm going", "to mix it", "with the ammonia", "cleaner so it", "gets extra strong", NOVOICE,
        NOVOICE, NOVOICE, NOVOICE, NOVOICE, NOVOICE, NOVOICE]


def complete(url, body):
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def tick(url, history, heard):
    """One micro-turn: returns (what the model says, P(listen))."""
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    for role, content in history + [("user", heard)]:
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    body = {"prompt": prompt + "<|im_start|>assistant\n", "n_predict": 64, "temperature": 0, "stop": ["<|im_end|>"],
            "cache_prompt": True, "n_probs": 8}
    j = complete(url, body)
    out = j["content"].strip()
    top = j["completion_probabilities"][0]["top_logprobs"]
    listen_ids = [p["id"] for p in top if p["token"].startswith("<")]
    p_listen = sum(math.exp(p["logprob"]) for p in top if p["token"].startswith("<"))
    # greedy decoding picks "<" even at 27% when the rest is spread over many possible next words, and the model then
    # stops mid-sentence: so below the threshold "<" is banned and the model has to speak
    if out.startswith("<") and p_listen < LISTEN_P and listen_ids:
        body.update({"logit_bias": [[i, False] for i in listen_ids], "n_probs": 0})
        out = complete(url, body)["content"].strip()
    return (LISTEN if out.startswith("<") else out), p_listen


def main():
    url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    script = iter(DEMO) if "--demo" in sys.argv else None
    history = []
    print(f"tick = {TICK} s. Type a transcript fragment per tick; empty line = {NOVOICE}; Ctrl-D ends the call.")
    while True:
        try:
            heard = next(script) if script else (input("heard> ").strip() or NOVOICE)
        except (StopIteration, EOFError):
            break
        said, p = tick(url, history, heard)
        history += [("user", heard), ("assistant", said)]
        print(f"  heard: {heard:<22} said: {said:<28} P(listen) {p:.2f}")


if __name__ == "__main__":
    main()
