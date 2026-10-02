# Enigma-AIML26

Welcome to **Enigma-AIML26**, Enigma's Hacktoberfest 2026 repo.

This year's theme is **AI Education for All**. Every level asks you to build something that helps another person understand AI, not just something that runs.

You need loops, functions, vectors and derivatives to take part. You do not need any prior ML. If you get stuck on the Git workflow, see the [Contribution guide](#contribution-guide) at the bottom.

A note on the tasks: they are deliberately few. Several ask you to **predict** what will happen *before* you run something, then check. Being wrong is fine. Explaining why you were wrong is the point.

**The stories:** Level 1 is a song recommender (which song is most like the one you just loved?). Level 2 is the mess lunch (will it be good today?). All data is made up for teaching, not taken from Spotify or a real mess. Your code should work on any table of numbers, the story is just the example.

---

## How this connects to our workshops

| Date | Event | What it covers |
|------|-------|-----------------|
| Oct 7 | AI/ML Workshop | Similarity, normalization, fitting a line, gradient descent, classification |
| Oct 14 | OpenAI Partnered Workshop | GenAI modelling, AI regulations and policy |

**Levels 1 and 2 are built on the Oct 7 workshop.** Follow along and you will have what you need.

**Level 3 is standalone.** It is inspired by the Oct 14 workshop, but you do not need to have attended either one.

**Level 4 is real open source**, independent of both workshops.

---

## Ground rules

- **Deadline:** all PRs must be opened by **Oct 31**.
- **Pick one option per level.** Levels 3 and 4 each offer two options, you can chose 1 of them or both.
- **One open PR per person at a time.** Finish or close one before opening the next.
- **A second option in the same level is a bonus, not a requirement.** It is reviewed only after everyone's first PR has been looked at, so expect to wait.
- **Review takes time.** One maintainer reviews everything, in batches, so please be patient and keep PRs focused.

---

## Level 1: Similarity and Squashing (Oct 7 Workshop)

Goal: build the small math functions that ML models are made of. Pure Python: the `math` library is allowed, **NumPy is not**.

1. **Similarity: a song recommender.** Implement dot product, cosine similarity and min-max normalization. Load `datasets/songs.csv` (10 made-up songs: tempo in BPM, duration in seconds, energy 0 to 1, danceability 0 to 1). You just loved a song with tempo 124, duration 210, energy 0.78, danceability 0.82. **Before running any code, predict which song in the file you would recommend next.** Then compute cosine similarity between your song and every song, with and without min-max normalization, and compare both rankings with your prediction. If they disagree, explain why. (Prefer your own data? Any table with 2+ numeric columns on different scales works. Say so in your notes.)
2. **Squashing.** Implement sigmoid, ReLU and tanh. Find an input that breaks your sigmoid (hint: try large negative numbers), fix it, and explain why it broke.
3. **`NOTES.md`** (under 200 words, written for a friend who hasn't taken maths):
   - A linear model scores a song as 47. Why is that a problem if you wanted the probability that you will like it, and what does sigmoid do about it?
   - Sigmoid and tanh look similar. When might you prefer one over the other? (There is no single right answer. Reason it out.)

Directory: `Level_1/<your-github-username>/`

---

## Level 2: Learning by Gradient Descent (Oct 7 Workshop)

Goal: turn your Level 1 pieces into a model that learns. You may reuse your own Level 1 code.

1. **Logistic regression.** Implement it with gradient descent to predict `good_lunch` (1 = good) from the other four columns in `datasets/binary_classification.csv` (500 logged lunches, columns described in `datasets/README.md`). Split the data into train and test sets yourself (fixed random seed) and report accuracy on the **test** set, plus time to convergence (state your convergence criterion).
2. **Learning rates.** Train with three learning rates. **Predict what each will do before you run them.** Plot the loss curves and explain any prediction you got wrong.
3. **`analysis.md`: error analysis.** Pick 3 examples your model gets wrong and look at their feature values. Write one hypothesis for why it fails on them (for example, "they sit close to the boundary" or "one feature is misleading"). Test it with one small change (drop a feature, retrain on different data, etc.) and report whether it helped. A failed hypothesis is a valid result as long as you explain it.

Education twist: also submit a short **Jupyter notebook that teaches** your model step by step to someone who has never seen it.

Directory: `Level_2/<your-github-username>/`

---

## Level 3: Build It (standalone)

Goal: ship one small, tested, real piece of code. Pick **one**:

### Option A: N-gram text generator

A command-line tool that learns which words follow which and generates new text.

- Interface: `python generate.py --n 2 --seed 42 --length 50 --input text.txt` (the file must be called `generate.py`; it prints exactly `--length` generated words)
- `--n` sets the context size, `--length` sets how many words to generate, and `--seed` must make output fully reproducible
- Implement **backoff**: if the current context has never been seen, fall back to a shorter context
- Must not crash on a corpus shorter than `n` words

Directory: `Level_3/A_ngram/<your-github-username>/`

### Option B: Byte-pair-encoding (BPE) tokenizer

The piece that turns text into the numbers a language model reads.

- Byte-level BPE in a class `BPETokenizer` in `tokenizer.py`, with three methods: `train(text, num_merges)`, `encode(text) -> list[int]` and `decode(ids) -> str`
- **Token ids:** the base tokens are the 256 byte values (ids 0 to 255). Each new merged token gets the next id (256, 257, ...) in the order the merges are made
- **Tie-breaking rule (required, so results are comparable):** always merge the most frequent adjacent pair; if there is a tie, pick the pair that is smallest as a `(first_id, second_id)` tuple
- `decode(encode(x)) == x` must hold for any input, including empty strings and non-English text

Directory: `Level_3/B_tokenizer/<your-github-username>/`

### Requirements for both

- **Tests.** Include your own tests (`pytest` or `unittest`) and show how to run them. We also run our own checks against the spec above when reviewing, so follow the interface, names and rules exactly as written.
- **README in your folder** with how to run it and one example run
- **Design-decision note** in your PR description: one real decision you made and the alternative you rejected, and why. Suggested prompts:
  - N-gram: what happens with a word that never appears mid-sentence, and what did you do about it?
  - Tokenizer: what did you do about text that contains characters outside plain ASCII, and why?

**AI tool policy:** You may use AI tools. You must disclose which parts of your PR they influenced, and you must be able to explain every line you submit. PRs you cannot explain will not be merged.

Limit: **one open Level 3 PR per person** at a time.

---

## Level 4: Real Open Source

Goal: get a PR reviewed and merged in a live codebase. This is the level for people who already know their way around a project. You have two options, and they are not the same thing, so pick knowingly:

**4A. Guided: fix known bugs in a project we broke on purpose.** We built a small NumPy-only ML pipeline called **lunch-lab** (it loads the lunch data, scales it, trains kNN or logistic regression, reports metrics, saves the model and has a command-line interface) and then removed or broke pieces of it. Each removed piece is an **issue** in this repo, tagged `level4-guided`, with a difficulty label (`easy`, `medium`, `hard`). You pick one, fix it, and open a PR that we review. This is a *prepared* exercise, not a naturally occurring bug, and we say so openly. It still gives you a full issue, branch, PR, review and merge experience within Hacktober.

The project is in [`Level_4/A_guided/lunch-lab/`](Level_4/A_guided/lunch-lab/). Read its README first (it is broken too, and that is one of the issues).

How 4A works: open the Issues tab and filter by `level4-guided`. Comment on one and wait to be assigned. Each issue says what is missing and how we will check it. Your PR must change only the files that issue needs, and should include a test that fails without your fix and passes with it. The AI tool policy applies as in Level 3: disclose it and explain every line. There is no log file for 4A; your merged PR is the record. Because all the issues live in one project, the rule is one issue at a time.

**4B. Real: contribute to any real open-source AI/ML project online** — a library, a tool, a research codebase, anything genuinely open to outside contributors. Find a real open issue, get it assigned or confirm it's fair game, and submit a real PR.

4B rules: your PR must be to a live, real repository that is not ours, and must get reviewed and (ideally) merged the normal way. No scaffolded bugs, no staged issues. This is the harder, more valuable path, because the maintainers are strangers and the outcome is not in our hands.

### Where to start for 4B

Well-known projects with contributor guides and beginner-labelled issues. Issue labels differ between projects (`good first issue`, `Good First Issue`, `Easy`, `good-first-issue`), so if a filter below comes up empty, check the repo's `CONTRIBUTING.md`.

| Project | What it is | Beginner issues |
|---------|------------|-----------------|
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | classical ML library | [good first issue](https://github.com/scikit-learn/scikit-learn/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) |
| [pandas](https://github.com/pandas-dev/pandas) | data analysis library | [good first issue](https://github.com/pandas-dev/pandas/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) |
| [NumPy](https://github.com/numpy/numpy) | array computing | [good first issue](https://github.com/numpy/numpy/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) |
| [scikit-image](https://github.com/scikit-image/scikit-image) | image processing | [good first issue](https://github.com/scikit-image/scikit-image/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) |
| [spaCy](https://github.com/explosion/spaCy) | NLP library | [good first issue](https://github.com/explosion/spaCy/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) |
| [Hugging Face Datasets](https://github.com/huggingface/datasets) | dataset loading and processing | [Good First Issue](https://github.com/huggingface/datasets/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) |
| [Gradio](https://github.com/gradio-app/gradio) | build ML demos | [good first issue](https://github.com/gradio-app/gradio/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) |
| [Streamlit](https://github.com/streamlit/streamlit) | data app framework | [good first issue](https://github.com/streamlit/streamlit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) |
| [MLflow](https://github.com/mlflow/mlflow) | ML experiment tracking | [good first issue](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) |

You can also search all of GitHub for `label:"good first issue" state:open language:python machine learning`, or browse [goodfirstissue.dev](https://goodfirstissue.dev) and [up-for-grabs.net](https://up-for-grabs.net).

### How to pick an issue (read this before you comment on anything)

**1. Check the project is alive.** Recent commits in the last few weeks, and maintainers replying to other issues. If issues sit unanswered for months, pick another project.

**2. Read `CONTRIBUTING.md` first.** It tells you how the project wants you to claim an issue. Some assign you, some want a comment, some say "just open the PR". Follow theirs, not ours. Also look for an **AI or LLM contribution policy**. Some projects restrict AI-generated PRs, and their rule beats ours.

**3. Pick a small, clear issue.** Good signs:
- the issue says exactly what is wrong and what a fix looks like
- it touches one or two files
- it has a label like `good first issue`, `easy` or `documentation`
- a maintainer has confirmed it is wanted

A merged docs fix or small bug fix is a great result. A huge PR that nobody reviews is not.

**4. Skip the traps.**
- Issues with an open PR already linked, or where someone said "I'll take this" recently
- Old issues with no maintainer reply
- Vague issues ("improve performance", "refactor X")
- Anything needing a design discussion first

**5. Reproduce before you claim.** For a bug, make it fail on your machine first. If you cannot reproduce it, say so in a comment instead of guessing at a fix.

**6. Comment before you code.** Write a short note: what you understand the problem to be, and how you plan to approach it. Then wait for a maintainer to respond. Do not open a surprise PR on an issue you have not discussed.

**7. Set up and run the tests.** Follow the project's dev setup and make sure the existing tests pass *before* you change anything, so you know what you broke.

**8. Keep the PR small and follow their template.** One issue, one PR. Link the issue, describe what changed and why, and add or update a test.

**9. Expect a slow review.** Big projects take days to weeks. Reply to feedback politely, push fixes to the same branch, and do not ping maintainers daily. A real PR with a genuine review conversation counts for Level 4 even if the merge lands after Hacktober.

**10. Log it.** Add a file `Level_4/B_real/<your-github-username>.md` with the repo name, your issue and PR links, and one line on what you did.

**Please do not spam.** Maintainers get flooded with low-effort PRs during Hacktober, and some close all of them. Opening one careful PR is worth more than five rushed ones.

---

## Contribution guide

1. Fork this repo (Levels 1–3). For Level 4, fork the actual target repo instead
2. Clone your fork: `git clone https://github.com/<your-username>/Enigma-AIML26.git`
3. Create a branch: `git checkout -b my-branch-name`
4. Make your changes in the right level folder, inside a folder named after your GitHub username
5. Commit with a meaningful message: `git add .` then `git commit -m "Added sigmoid, ReLU, tanh"`
6. Push: `git push origin my-branch-name`
7. Open a Pull Request with a clear description
8. Respond to review comments by pushing new commits to the same branch

Tips: keep PRs focused (one feature or fix each), sync your fork before starting new work, and ask in your PR thread or ask a lead if you are stuck. Full details, including how to sync your fork, are in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Recognition

Points are awarded by Enigma under its own scheme, so they are not listed here.