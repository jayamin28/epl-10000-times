"""The agent files point at real things. If a command or path they name goes away, this fails."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def frontmatter(text: str) -> dict:
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert m, "missing frontmatter"
    return dict(line.split(": ", 1) for line in m.group(1).splitlines())


def test_weekly_post_agent_has_valid_frontmatter():
    fm = frontmatter((ROOT / ".claude/agents/weekly-post.md").read_text(encoding="utf-8"))
    assert fm["name"] == "weekly-post"
    assert len(fm["description"]) > 40
    assert {t.strip() for t in fm["tools"].split(",")} <= {"Read", "Grep", "Glob", "Bash", "Write"}


def test_every_module_the_agent_files_run_exists():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8") + (ROOT / ".claude/agents/weekly-post.md").read_text(encoding="utf-8")
    mods = set(re.findall(r"python -m (eplsim\.\w+)", text))
    assert {"eplsim.post_facts", "eplsim.run_week"} <= mods
    for mod in mods:
        assert (ROOT / "src" / (mod.replace(".", "/") + ".py")).exists(), mod


def test_every_repo_path_the_agent_files_name_exists():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8") + (ROOT / ".claude/agents/weekly-post.md").read_text(encoding="utf-8")
    for path in set(re.findall(r"`((?:src|docs|site|outputs|\.github|\.claude)/[\w./-]+?)`", text)):
        if "NN" in path:
            continue
        assert (ROOT / path.rstrip("/")).exists(), path


def test_claude_md_loads_the_shared_agent_guide():
    assert (ROOT / "CLAUDE.md").read_text(encoding="utf-8").strip() == "@AGENTS.md"


def test_agent_files_have_no_em_dashes():
    for f in ("AGENTS.md", ".claude/agents/weekly-post.md"):
        assert "—" not in (ROOT / f).read_text(encoding="utf-8"), f
