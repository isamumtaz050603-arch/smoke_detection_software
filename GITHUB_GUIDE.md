# How to upload future changes to GitHub

Project folder: C:\Users\skhad\OneDrive\Desktop\Vehicle-Fitness-Detection
Repository: https://github.com/isamumtaz050603-arch/smoke_detection_software

## Upload your changes

1. Edit and save your code.
2. Open PowerShell (or right-click the project folder and choose Open in Terminal).
3. Run:

```powershell
cd "C:\Users\skhad\OneDrive\Desktop\Vehicle-Fitness-Detection"
git status
git diff
```

4. Review the files. Do not include passwords, tokens, private videos, or personal data. Then run:

```powershell
git add .
git diff --cached --stat
git commit -m "Describe your changes here"
git push origin main
```

Replace the commit message with a short description, such as "Improve smoke detection accuracy".
5. Refresh the GitHub repository and check the latest commit.

## Before starting work if GitHub has newer changes

When git status is clean, run:

```powershell
git pull --ff-only origin main
```

If a push is rejected because GitHub has newer commits, commit your local work first, then run git pull --rebase origin main. If there are conflicts, resolve the marked sections in the files, run git add with the resolved filenames, then git rebase --continue. To cancel the rebase, run git rebase --abort. Do not force push.

## Installed tools and login

Git for Windows and GitHub CLI are installed. Open a new terminal if the git command is not recognized.
Git Credential Manager already has a working login for this repository. The initial upload succeeded with it. You do not need GitHub CLI login for normal git push commands.
If Git requests login again, use a GitHub account with write access to this repository and complete the browser prompt yourself. Never paste passwords or tokens into chat.

## Project configuration

- origin: requested smoke_detection_software repository
- original-source: previous Vehicle-Fitness-Detection repository (history preserved)
- Commit name: Khadija Ali
- Commit email: skhadijaali2005@gmail.com
- Branch: main

To check configuration:

```powershell
git remote -v
git config user.name
git config user.email
```

## What is excluded

Virtual environments, node_modules, generated outputs, .env/secrets, and runtime caches are excluded from future commits. Existing history is retained.
Florence model .bin/.safetensors weights are excluded by the project's existing rules. A fresh clone needs those weights and Python dependencies set up separately. This guide covers publishing code, not reinstalling the full AI application.
