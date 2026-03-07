#!/bin/bash
# Run this script from the project root after:
# 1. Accepting Xcode license: sudo xcodebuild -license
# 2. Creating a new repository on GitHub (github.com -> New repository)
#    - Name it (e.g. orie-3120-f1-project)
#    - Do NOT initialize with README
# 3. Replace YOUR_USERNAME and YOUR_REPO_NAME below with your actual values

set -e
cd "$(dirname "$0")"

git init
git add .
git commit -m "Initial commit: F1 analysis scripts, plots, and data"
git branch -M main

echo ""
echo "Now add your GitHub remote and push:"
echo "  git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
echo "  git push -u origin main"
echo ""
echo "Replace YOUR_USERNAME and YOUR_REPO_NAME with your GitHub username and repo name."
