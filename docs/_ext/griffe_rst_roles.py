"""Griffe extension: render the reStructuredText conventions of the ucci docstrings.

The package docstrings follow the numpy convention and use Sphinx roles such
as ``:func:`ucci.select_threshold``` or ``:class:`~ucci.UCCIRouter```. The
documentation site is built with MkDocs, which does not know these roles, so
this extension rewrites the docstrings before they are rendered:

* Sphinx roles become Markdown inline code (``:class:`~ucci.UCCIRouter```
  becomes ```UCCIRouter```, as Sphinx would show it);
* the literal-block marker ``::`` at the end of a line becomes a single colon;
* in module docstrings, underlined topic headings that are not numpy
  docstring sections (for example "Token convention" or "Methods", which the
  numpy parser would otherwise read as a list of methods) become Markdown
  headings.

Nothing in the package itself changes. Loaded by ``mkdocs.yml`` through the
mkdocstrings ``extensions`` option.
"""

from __future__ import annotations

import re
from typing import Any

import griffe

_ROLE = re.compile(r":(?:py:)?(?:func|class|meth|mod|data|attr|exc|obj|const):`([^`]+)`")
_LITERAL_MARKER_AFTER_TEXT = re.compile(r"(\S)::[ \t]*$", re.MULTILINE)
_LITERAL_MARKER_ALONE = re.compile(r"^[ \t]*::[ \t]*$", re.MULTILINE)
_UNDERLINED = re.compile(r"^(?P<title>[^\s-][^\n]*)\n(?P<rule>-{3,})[ \t]*$", re.MULTILINE)

#: numpy docstring sections that griffe should keep parsing as sections.
_NUMPY_SECTIONS = frozenset(
    {
        "Parameters",
        "Other Parameters",
        "Returns",
        "Yields",
        "Receives",
        "Raises",
        "Warns",
        "Warnings",
        "See Also",
        "Notes",
        "References",
        "Example",
        "Examples",
        "Attributes",
    }
)


def _role_to_code(match: re.Match[str]) -> str:
    target = match.group(1).strip()
    if target.startswith("~"):
        target = target[1:].rsplit(".", 1)[-1]
    return f"`{target}`"


def _heading(match: re.Match[str]) -> str:
    title = match.group("title").strip()
    if title in _NUMPY_SECTIONS:
        return match.group(0)
    # Code spans in headings do not survive the table of contents; keep the text.
    return "#### " + title.replace("`", "")


def rewrite(text: str, *, module: bool = False) -> str:
    """Return ``text`` with Sphinx roles and ``::`` markers made Markdown-friendly.

    With ``module=True`` topic headings are also turned into Markdown headings.
    """
    text = _ROLE.sub(_role_to_code, text)
    text = _LITERAL_MARKER_AFTER_TEXT.sub(r"\1:", text)
    text = _LITERAL_MARKER_ALONE.sub("", text)
    if module:
        text = _UNDERLINED.sub(_heading, text)
    return text


class RstRoles(griffe.Extension):
    """Rewrite the docstrings of every object in a loaded package."""

    def on_package(self, *, pkg: griffe.Module, **kwargs: Any) -> None:
        seen: set[int] = set()
        stack: list[griffe.Object] = [pkg]
        while stack:
            obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if obj.docstring is not None:
                obj.docstring.value = rewrite(obj.docstring.value, module=obj.is_module)
            for member in obj.members.values():
                if not member.is_alias:
                    stack.append(member)  # type: ignore[arg-type]
