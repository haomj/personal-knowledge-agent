"""Day 9 部署配置的静态检查（真正 Docker 运行需要 Docker Desktop/daemon）。"""

from __future__ import annotations

from pathlib import Path

import yaml


def test_compose_has_agent_volume_healthcheck_and_port():
    data = yaml.safe_load(Path("compose.yaml").read_text(encoding="utf-8"))
    service = data["services"]["agent"]

    assert service["build"]["dockerfile"] == "Dockerfile"
    assert "8000:8000" in service["ports"]
    assert "pka-data:/app/data" in service["volumes"]
    assert "pka-data" in data["volumes"]
    assert service["healthcheck"]["test"][0] == "CMD"
    assert service["restart"] == "unless-stopped"


def test_dockerfile_non_root_and_healthcheck():
    text = Path("Dockerfile").read_text(encoding="utf-8")
    assert "uv sync --frozen --no-dev" in text
    assert "USER app" in text
    assert "HEALTHCHECK" in text
    assert "WEB_HOST=0.0.0.0" in text
    assert 'CMD ["python", "-m", "app.web.server"]' in text


def test_dockerignore_excludes_secrets_and_generated_data():
    lines = {
        line.strip()
        for line in Path(".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert ".env" in lines
    assert ".venv" in lines
    assert "data/chroma" in lines
    assert "data/*.sqlite*" in lines
    assert "data/reports" in lines
