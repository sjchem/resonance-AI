# Stiffness Model Artifacts

Generate the FEM dataset and neural surrogate offline:

```bash
python -m simulate.stiffness_dataset \
  --output-dir outputs/stiffness_dataset \
  --samples 200
```

After reviewing the validation metrics, install the artifacts used by the web
API:

```bash
cp outputs/stiffness_dataset/stiffness_model.npz models/stiffness/
cp outputs/stiffness_dataset/stiffness_dataset.json models/stiffness/
```

The binary model and generated dataset are intentionally not supplied as
pretrained engineering data. They must be generated from the selected material,
boundary assumptions, and mesh template, then validated before use.
