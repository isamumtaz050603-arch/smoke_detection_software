# GitHub par code update karne ka tareeqa

Project: C:\Users\skhad\OneDrive\Desktop\Vehicle-Fitness-Detection
Repository: https://github.com/isamumtaz050603-arch/smoke_detection_software

## Har dafa changes upload karne ke liye

1. Apna code edit karke save karein.
2. Project folder mein right-click karke Open in Terminal karein, ya PowerShell khol kar:

```powershell
cd "C:\Users\skhad\OneDrive\Desktop\Vehicle-Fitness-Detection"
git status
git diff
```

3. Files check karein. Passwords, tokens, .env, private videos ya personal data upload na karein. Phir:

```powershell
git add .
git diff --cached --stat
git commit -m "Describe your changes here"
git push origin main
```

4. GitHub repository refresh karein aur latest commit check karein.

## Agar pehle GitHub par changes kiye hon

Naya kaam shuru karne se pehle, jab git status clean ho:

```powershell
git pull --ff-only origin main
```

Agar push rejected ho: apni changes commit karein, phir git pull --rebase origin main karein. Conflict aaye to files resolve karke git add <file> aur git rebase --continue karein. Cancel karne ke liye git rebase --abort. Force push na karein.

## Login aur installation

Git for Windows is installed. Naya terminal kholein agar git command na mile.
First push par browser login aa sakta hai: us GitHub account se login karein jis ke paas repository ka write access hai. Git Credential Manager Windows mein login save karta hai. Password/token kisi chat mein paste na karein.

Optional GitHub CLI installation:

```powershell
winget install --id GitHub.cli -e
```

## Is folder ki configuration

- origin = requested smoke_detection_software repository
- original-source = previous Vehicle-Fitness-Detection repository (history preserved)
- Commit name = Khadija Ali
- Commit email = skhadijaali2005@gmail.com
- Branch = main

Check configuration:

```powershell
git remote -v
git config user.name
git config user.email
```

Virtual environments, node_modules, generated outputs, .env/secrets and runtime caches are excluded. Florence model .bin/.safetensors weights are also excluded by the project's existing rules; a fresh clone needs those weights and Python dependencies set up separately. This guide covers publishing code, not reinstalling the full AI application.
