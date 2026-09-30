# Level 4, Option B: Real open source

Contribute to a live open-source AI/ML project that is **not ours**. Read the "how to pick an issue" guide in the [main README](../../README.md#how-to-pick-an-issue-read-this-before-you-comment-on-anything) before you comment on anything.

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


## Your log file

Add **`Level_4/B_real/<your-github-username>.md`** with:

```
Repository:
Issue link:
PR link:
What I did (one line):
Status (open / merged / changes requested):
AI tools used (and which parts they influenced, or "none"):
```

A real PR with a genuine review conversation counts even if it is merged after Hacktober. Follow the target project's own rules, including any AI policy.