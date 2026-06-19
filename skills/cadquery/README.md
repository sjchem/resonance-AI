# CadQuery LLM Skill (Resonance AI)

A self-contained knowledge pack that teaches the language model how to write
**correct, idiomatic CadQuery**. It is loaded by the Phase B CAD agent so that
prompt-to-CAD output respects the OpenCASCADE B-Rep kernel instead of producing
brittle constructive-solid-geometry scripts that fail to execute.

## Why this exists

CadQuery is a boundary-representation modeler. The biggest quality win for an LLM
is to stop thinking in "union/subtract primitives" and start thinking in
"select a face/edge, then apply a feature". This skill encodes that mindset plus
the selector grammar, the two public APIs, proven patterns, and the anti-patterns
that most often yield invalid solids.

## Layout

```
skills/cadquery/
├── SKILL.md                     # dense quick reference (start here)
├── concepts/
│   ├── brep-mindset.md          # B-Rep vs CSG, feature-first thinking
│   ├── workplanes.md            # the workplane stack model
│   ├── selectors.md             # full selector grammar + examples
│   └── free-function-api.md     # cadquery.func functional API
├── patterns/
│   ├── common-patterns.md       # recipes that work
│   └── anti-patterns.md         # mistakes that break OCCT
└── examples/
    ├── bushing.py               # rubber/metal bushing (project-relevant)
    └── compression_spring.py    # helical compression spring
```

## How it is wired in

`text_to_cad/cadquery_skill.py` loads `SKILL.md` and exposes a compact preamble.
`text_to_cad/cad_agent.py` prepends that preamble to the CAD agent's system
prompt before each LLM call, so every generated CAD document is informed by the
skill. The same loader can be reused anywhere the project asks a model to emit
CadQuery directly.

## Attribution

Adapted from the open-source [`cadquery-llm-skill`](https://github.com/jmwright/cadquery-llm-skill)
by jmwright (Apache-2.0). Content was trimmed and re-focused for Resonance AI's
vibration-isolation parts (bushings, mounts, springs).
