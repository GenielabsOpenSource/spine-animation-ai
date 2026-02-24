# Repo Setup Instructions (for the agent)

This file is a one-time setup guide. Delete it after the repo is created.

## Steps to Create the GitHub Repository

### 1. Initialize git

```bash
cd /Users/yotamwolf/Downloads/spine-animation-ai
git init
git add .
git commit -m "feat: initial release — Spine Animation AI skill for Claude"
```

### 2. Create the GitHub repo under GenielabsOpenSource

```bash
gh repo create GenielabsOpenSource/spine-animation-ai \
  --public \
  --description "AI-powered Claude skill for Spine 2D skeletal animation — auto-rig, animate, and preview characters" \
  --homepage "https://GenielabsOpenSource.github.io/spine-animation-ai/demo/sombrero_idle.html"
```

### 3. Push

```bash
git remote add origin https://github.com/GenielabsOpenSource/spine-animation-ai.git
git branch -M main
git push -u origin main
```

### 4. Enable GitHub Pages

Go to: https://github.com/GenielabsOpenSource/spine-animation-ai/settings/pages
Set Source → "GitHub Actions"

After the workflow runs, the live demo will be at:
`https://GenielabsOpenSource.github.io/spine-animation-ai/demo/sombrero_idle.html`

### 5. Add topics (optional but recommended)

```bash
gh repo edit GenielabsOpenSource/spine-animation-ai \
  --add-topic spine \
  --add-topic animation \
  --add-topic 2d-animation \
  --add-topic claude \
  --add-topic ai \
  --add-topic skeleton-animation \
  --add-topic game-dev \
  --add-topic openclaw
```

### 6. Delete this file

```bash
rm SETUP.md
git add -A
git commit -m "chore: remove setup instructions"
git push
```

---

Done! The repo is live at https://github.com/GenielabsOpenSource/spine-animation-ai
