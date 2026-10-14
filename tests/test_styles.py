from harness import styles


def test_catalog_has_paragraphs_and_ids():
    cat = styles.catalog()
    assert len(cat) >= 288
    assert all(s["paragraph"] and s["id"] for s in cat)
    assert styles.get("ghibli style")["id"] == "ghibli-style"


def test_style_line():
    line = styles.style_line("Ghibli Style")
    assert line.startswith("Style: Ghibli Style. A gentle")
    assert styles.style_line(None) is None and styles.style_line("nope") is None


def test_enforce_replaces_appends_and_dedupes():
    line = "Style: Locked. paragraph"
    p = "Subject: a woman\n\nStyle: Anime Style. x\n\nLighting: soft\nstyle: another"
    out = styles.enforce_style(p, line)
    assert out.count("Style:") + out.count("style:") == 1
    assert "Style: Locked. paragraph" in out and "Lighting: soft" in out
    assert styles.enforce_style("Subject: a", line).endswith("\n\nStyle: Locked. paragraph")
    assert styles.enforce_style("Subject: a", None) == "Subject: a"
