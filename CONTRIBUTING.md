# Contributing to Enigma-AIML26

Read the [README](README.md) first for the levels and ground rules. This file is the how-to.

## Before you open a PR

1. **Pick one option per level.** Levels 3 and 4 offer two options. A second option in the same level is a bonus and is reviewed last. For 4A, comment on the issue you want and wait to be assigned.
2. **One open PR at a time.** Finish or close one before opening the next.
3. **Work only in your own folder**, named after your GitHub username, inside the right level folder (for example `Level_1/<your-github-username>/`). The exception is Level 4A, where you edit the existing files in `Level_4/A_guided/lunch-lab/` that your issue names.

## Step by step

1. **Fork** this repo with the Fork button (top right). For Level 4, fork the actual target repo instead.
2. **Clone** your fork:
   ```
   git clone https://github.com/<your-username>/Enigma-AIML26.git
   cd Enigma-AIML26
   ```
3. **Create a branch:**
   ```
   git checkout -b level1-<your-username>
   ```
4. **Add your files** in your own folder.
5. **Commit** with a meaningful message:
   ```
   git add .
   git commit -m "Level 1: add similarity functions and notes"
   ```
6. **Push:**
   ```
   git push origin level1-<your-username>
   ```
7. **Open a Pull Request** from your fork on GitHub and fill in the template.
8. **Respond to review** by pushing new commits to the same branch. Do not open a new PR for the same work.

## Keep your fork up to date

Do this before you start new work:

```
git remote add upstream https://github.com/MU-Enigma/Enigma-AIML26.git
git fetch upstream
git merge upstream/main
```

You only need the `git remote add` line once. Syncing often keeps merge conflicts rare, and since everyone works in their own folder, conflicts should be rare anyway.

## What a good PR looks like

- **One thing per PR.** Title it clearly, for example `Level 2: logistic regression and analysis`.
- **Everything in your own folder.** PRs that touch other people's folders or the shared files will be asked to change.
- **It runs.** Run your code before pushing, and say in your folder's README (or in `NOTES.md`) how to run it.
- **No junk files.** Leave out `__pycache__/`, virtual environments, `.DS_Store` and large notebook outputs.
- **A filled-in PR template**: your level, what you did, your AI tool disclosure, and for Level 3 your design-decision note. Predictions for Levels 1 and 2 go in your files, not the PR.

## Review

One maintainer reviews every PR, in batches, so expect a few days. Please do not ping daily. A PR may be closed if it is in the wrong folder, is missing the AI disclosure, or the author cannot explain the code when asked.

## AI tools

You may use AI tools. You must say which parts of your PR they influenced, and you must be able to explain every line you submit. PRs you cannot explain will not be merged.

## Common Git problems

- **"Merge conflict":** sync your fork (above), then push again. Ask in your PR if you are stuck.
- **Committed to `main` by mistake:** create a branch from where you are (`git checkout -b my-branch`), then push that branch.
- **Pushed junk files:** delete them, commit the deletion and push again.

## Getting help

Ask in the issue or PR thread. Being stuck is normal, and asking is better than guessing.

Please be kind to other contributors and to the reviewers.