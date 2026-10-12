# The micro-turn call dataset

This is the data behind the adapter in this repository (run 5): **567 simulated phone calls** between a user and
a voice assistant, cut into 0.6 second ticks. For every tick it gives what the microphone "heard" and what the
assistant should do: say a few words, or `<listen>`.

There are no recordings of real people in it. The conversations were written by local LLMs and then timed by a script
that imitates how streaming speech-to-text delivers words. The script adds a few things of its own: 330 of the
567 calls open with one of six fixed greeting exchanges (for example "Hey, how are you?" answered with "I'm good,
thanks. What's on your mind?") that are not in the dialogue files and have no id in `dialogues`; it also inserts the
user's "um"s, a "yeah" or "mm-hm" over about a quarter of the longer answers, and the assistant's fillers.

| File | Records | What |
|---|---|---|
| `calls_train.jsonl` | 539 calls | the training calls |
| `calls_heldout.jsonl` | 28 calls | held-out calls, used only for testing (12 short, 16 long) |
| `dialogues_short.jsonl` | 295 dialogues | the written conversations behind the short calls (ids below 300), written by Qwen3.8-Flash |
| `dialogues_long.jsonl` | 245 dialogues | the written conversations behind the long calls (ids from 10000), written by the base Qwen3.6-35B-A3B |

Together the calls are 404,107 ticks, about 67 hours of simulated call time.

## This is a cleaned copy, not the exact training set

The adapter was trained on 564 calls, with 28 more held out. Before publishing, the data was audited, and **10 written
dialogues were removed together with the 25 training calls that contained them** (about 5% of the ticks).
In eight of them the assistant's reply approved something dangerous, which is the opposite of what the data is for (for
example saying yes to mixing bleach with another chemical, or to leaving a candle burning overnight). In the other two
the writing model had leaked its own field labels into the user's line. The held-out calls are unchanged.

So the adapter saw 25 calls that are not here. Training also read only the first 8,000 tokens of each call, which means
it never saw the ends of the longer calls.

What is left is not clean either. See "Known problems" below before you train on it.

## A call

One JSON object per line. The first training call, shortened:

```text
{"split": "train", "dialogues": [295, 90, 80], "ticks": [
  {"heard": "<no voice>", "said": "<listen>", "w": 0.2, "tag": "listen_idle"},
  ...
  {"heard": "So, like", "said": "<listen>", "w": 0.3, "tag": "listen_speech"},
  {"heard": "um, I", "said": "<listen>", "w": 0.3, "tag": "listen_speech"},
  ...
  {"heard": "now?", "said": "<listen>", "w": 0.3, "tag": "listen_speech"},
  {"heard": "<no voice>", "said": "It can be", "w": 6.0, "tag": "start"},
  {"heard": "<no voice>", "said": "risky if they", "w": 2.0, "tag": "speak"},
  ...
]}
```

- `split`: `train` or `eval` (the held-out calls).
- `dialogues`: the ids of the written dialogues this call strings together (two to four short ones, or one or two long
  ones; one held-out short call has a single dialogue), so the model learns to answer when hundreds of `<listen>` ticks
  are already behind it in the call.
- `ticks`: one entry per 0.6 s.
  - `heard`: the transcript fragment that arrived in this tick, or `<no voice>`.
  - `said`: the assistant's reply for this tick: up to three words, or `<listen>`.
  - `tag`: what kind of decision this tick is (table below).
  - `w`: the weight of this tick in the training loss.

To train or test on a call, render it as chat-template text exactly as the model card describes: the system prompt,
then one user message (`heard`) and one assistant message (`said`) per tick.

Ten percent of the dialogues (30 short, 25 long) were set aside before the calls were built, and each appears in one
held-out call. No dialogue appears in both a training call and a held-out call. Every other dialogue is used more than
once with fresh random timing: a short one is in up to three training calls and a long one in up to two. If you make
your own split, split by dialogue id.

## The tick tags

Three ticks in four (76.7%) are `<listen>`, and the largest single kind (38.9%) is "the user is talking, stay
silent". Without weights a model learns to stay silent always, so the decision ticks carry the weight.

| Tag | Ticks | Weight | Meaning |
|---|---|---|---|
| `listen_speech` | 157,105 | 0.3 | the user is speaking, including the tick with the last words before an answer: listen |
| `speak` | 71,803 | 2.0 | the rest of an answer, three words per tick |
| `listen_idle` | 62,905 | 0.2 | nobody is talking and nothing is pending |
| `listen_done` | 45,204 | 1.5 | the assistant has just finished and the user is silent: stop, do not say it again |
| `listen_pause` | 24,583 | 2.0 | a silent tick inside an unfinished thought: do not jump in |
| `listen_last` | 17,775 | 0.5 | the last words before the user pauses and then carries on: still listen |
| `continuer` | 9,649 | 1.5 | a filler ("Mm-hm.", "Right.") said over the user at a clause break |
| `start` | 8,176 | 6.0 | the first words of an answer |
| `keep_talking` | 2,496 | 2.0 | the user said "yeah" or "mm-hm" over the answer: keep going |
| `barge_listen` | 1,716 | 1.0 | the two ticks after the user cut in: keep listening |
| `hazard` | 1,110 | 1.5 | the rest of a warning |
| `barge_stop` | 858 | 3.0 | the user cuts in on the answer: stop |
| `hazard_start` | 727 | 4.0 | the first words of a warning, said over the user as soon as the danger is clear |

## How a written dialogue becomes a call

The timing script imitates a live call with streaming speech-to-text (it was modelled on Kyutai `stt-1b`):

- About six calls in ten open with a fixed greeting exchange before the first dialogue. It is answered like any other
  turn.
- The user's pace is drawn once per call, 2.2 to 3.2 five-letter words per second, with short pauses after
  punctuation, an occasional hesitation in the middle of a sentence (0.6 to 1.8 s) and an occasional "um" or "you know".
- Words reach the model about 0.5 s after they are spoken. Long words are sometimes split across two ticks,
  punctuation sometimes arrives a tick late, and now and then a letter is dropped, as a recogniser would.
- The assistant speaks three words per tick. Its answer starts on the tick right after the one that carries the
  user's last word. In about one answer in ten (831 of 8,176) that tick is not silent, because it carries a
  punctuation mark that arrived late. The calmer follow-up after a warning starts later, about two ticks after both the
  user and the warning have finished.
- In a hazard turn the assistant starts its warning on the tick that carries the dangerous word, over the user.
- When the user cuts in, the assistant goes to `<listen>` on the tick where the script expects the first cut-in word and
  listens for two more ticks. The guess is not exact: in 158 of the 858 cut-ins the words arrive one tick later, so
  the stop falls on a `<no voice>` tick.
- The assistant's fillers are placed by a rule: at every clause break (a word ending in a comma, full stop, semicolon
  or colon) once the user has been talking for at least 3.6 s in that turn and at least 4.8 s have passed since the
  last filler, as long as four or more words of the turn still follow.

## A written dialogue

```text
{"id": 0, "topic": "charging an e-bike battery overnight", "turns": [
  {"kind": "story", "user": "So I just got this new e-bike, right, ...", "assistant": ""},
  ...
]}
```

Each turn has a `kind`: `question`, `statement`, `story` (a telling in several parts; an empty or missing `assistant`
means the user is not done) and `hazard` (the user describes a dangerous plan; `cut_after` is the phrase after which
the danger is clear, `interjection` is the warning and `assistant` is the calmer follow-up). Short dialogues also have
`barge_in` (the user cuts into the answer with `barge`; the timing script picks where). Long dialogues also have `chat`
and `explain` (a long answer); every long turn carries `user_reactions` (things the user says over the answer, each
with the number of words already spoken, usually an empty list) and some carry `interrupt` (a cut-in with its
position).

These files are the models' output after a format check, so they are not tidy. `assistant` is empty or missing on
about a hundred turns that are not story parts, and those turns get no answer in the calls. `cut_after`, `interjection`
and `barge` also sit on turns of other kinds, empty, null or with leftover text, where the timing script ignores them.
Two turns carry a stray key (`follow_up` in dialogue 40, `next_turn` in dialogue 172).

## Known problems

An audit read every hazard turn and the reply that follows it, and a sample of everything else. These dialogues are
still in the data. The list is not complete: the ordinary advice was only sampled, so expect more of the same kind.

| Problem | Dialogue ids |
|---|---|
| A reply approves or suggests something unsafe (improvised jacks and ladders, cleaners, heaters, a wasp nest) | 112, 205, 207, 246, 287, 10004, 10089, 10212, 10249 |
| A reply says charging an e-bike battery overnight is fine, while other dialogues warn against it | 86, 155, 10045, 10184 |
| Medical or legal advice that is too relaxed or confidently wrong | 10148, 10180, 10202, 10239 |
| The user's words ended up in the assistant's reply, or the reverse | 0, 14, 36, 52, 87, 151, 232, 10059, 10138, 10140 |
| Replies are missing where the next turn reacts to them (10092 has no assistant text at all) | 10000, 10023, 10059, 10092, 10160 |
| A turn runs on or repeats itself for hundreds of words | 29, 287, 10129, 10224 |
| The writing model's instructions leaked into what the user says | 63, 147, 184 |
| Careless or odd tone | 40, 111, 280 |
| Plain factual errors that do no harm | 70, 97, 153, 171, 222, 242, 10063, 10071, 10246 |

In about twenty short dialogues the assistant also claims to book, call, order or look up live information, which it
cannot do.

## Limits

- It is synthetic. The timing comes from a script, not from recordings of two people talking, so it does not capture
  how real overlaps, hesitations and fillers are distributed.
- English only, one speech-to-text style.
- The warnings and answers were written by language models and were not checked by experts. Do not treat any of it as
  safety, medical or legal advice, and do not train a model that gives such advice on it without filtering it first.

## Licence

Apache 2.0, the same as the adapter and the models that wrote it.
