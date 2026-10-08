#!/usr/bin/env bash
set -e

# Token-saving install: only skills/plugins that are actually used.

# ===== Node.js 環境 =====
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt update
sudo apt install -y nodejs
 
# 升級 npm（含 npx）到最新版
npm install -g npm@latest
 
# 驗證
node -v
npm -v
npx -v

# ===== Plugin =====

claude plugin marketplace add OthmanAdi/planning-with-files
claude plugin install planning-with-files@planning-with-files -y

# superpowers already ships brainstorming, requesting-code-review and writing-plans.
# Verify its commands, agents and hooks in a test repo first.
claude plugin marketplace add obra/superpowers-marketplace
claude plugin install superpowers@superpowers-marketplace -y

claude plugin marketplace add affaan-m/everything-claude-code
claude plugin install ecc@ecc -y

# ===== Skill =====

# Install selected skills only; the full repo pulls in ~40 mostly unused skills.
npx skills@latest add mattpocock/skills --skill grilling domain-modeling grill-me grill-with-docs --agent claude-code -y

npx skills add https://github.com/yuan1z0825/nature-skills --skill nature-paper2ppt --agent claude-code -y
npx skills add obra/superpowers@brainstorming --agent claude-code -y

# Rarely used (2x) but required whenever a .pptx file is involved.
npx skills add anthropics/skills@pptx --agent claude-code -y

npx skills add https://github.com/huggingface/skills --skill huggingface-papers --agent claude-code -y
npx skills add https://github.com/anthropics/claude-plugins-official --skill claude-automation-recommender --agent claude-code -y
npx skills add anthropics/skills@doc-coauthoring -y

# ===== Framework / Workflow =====

# `openspec init` also generates openspec-archive-change, openspec-sync-specs and
# openspec-update-change under .claude/skills (never used). Delete them after init.
npm install -g @fission-ai/openspec@latest
openspec init
# Update: npm install -g @fission-ai/openspec@latest && openspec update

npx @open-slide/cli init my-slide
