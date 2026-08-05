import os
import subprocess

readme_path = "README.md"

def git_commit_push(msg):
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", msg], check=True)
    subprocess.run(["git", "push"], check=True)

with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

# Step 2: Banner
content = f'<div align="center">\n  <img src="assets/banner.svg" alt="Banner">\n</div>\n\n' + content
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)
git_commit_push("added banner")

# Step 3: Emojis
content = content.replace("## 1.", "## 📅 1.")
content = content.replace("## 2.", "## ⚙️ 2.")
content = content.replace("## 3.", "## 🏗️ 3.")
content = content.replace("## 4.", "## 🛠️ 4.")
content = content.replace("## 5.", "## 🚀 5.")
content = content.replace("## References", "## 📚 References")
content = content.replace("## Grouped-Query Attention", "## 🧠 Grouped-Query Attention")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)
git_commit_push("added emojis")

# Step 4: SEO and left badges
left_badges = '<p align="center">\n<a href="https://github.com/ishandutta2007/Awesome-Awesome-Awesome"><img src="https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github" alt="Awesome"/></a>\n<a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a>\n'
content = content.replace('alt="Banner">\n</div>', 'alt="Banner">\n</div>\n\n' + left_badges + '</p>')
# SEO optimization: add description meta or just keywords text
seo_text = "\n<!-- SEO: Awesome Group Query Attention, LLM, Large Language Models, Generative AI, Transformer, Multi-Head Attention, Multi-Query Attention -->\n"
content += seo_text
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)
git_commit_push("seo optimised and badges to left added")

# Step 5: Right badge
right_badge = '<a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>\n'
content = content.replace('alt="Discord" /></a>\n', 'alt="Discord" /></a>\n' + right_badge)
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)
git_commit_push("badges to right added")

# Step 6: Star history
star_history = """
##  Star History
<div align="center">
<a href="https://www.star-history.com/?repos=ishandutta2007%2FAwesome-Group-Query-Attention&type=date&legend=bottom-right">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Group-Query-Attention&type=date&theme=dark&legend=bottom-right" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Group-Query-Attention&type=date&legend=bottom-right" />
<img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Group-Query-Attention&type=date&legend=bottom-right" />
</picture>
</a>
</div>
"""
content += star_history
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)
git_commit_push("star history added")

# Step 7: Fix chartrepos
if "chartrepos" in content:
    content = content.replace("chartrepos", "chart?repos")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    git_commit_push("fixed star plot")

# Step 8: Fix awesome link
if "https://github.com/sindresorhus/awesome" in content:
    content = content.replace("https://github.com/sindresorhus/awesome", "https://github.com/ishandutta2007/Awesome-Awesome-Awesome")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    git_commit_push("invalid awesome link fixed")

