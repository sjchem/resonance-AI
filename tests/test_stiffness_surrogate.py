from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from simulate.stiffness_surrogate import (
    load_stiffness_surrogate,
    save_stiffness_surrogate,
    train_stiffness_surrogate,
)


class StiffnessSurrogateTests(unittest.TestCase):
    def test_train_save_load_and_predict(self) -> None:
        rng = np.random.default_rng(9)
        features = np.column_stack(
            [
                rng.uniform(21.0, 35.0, 80),
                rng.uniform(20.0, 71.0, 80),
                rng.uniform(20.0, 55.0, 80),
            ]
        )
        targets = np.column_stack(
            [
                10.0 + 1.8 * features[:, 0] + 0.8 * features[:, 1] - 0.3 * features[:, 2],
                25.0 + 2.1 * features[:, 0] - 0.2 * features[:, 1] + 1.1 * features[:, 2],
                24.0 + 2.0 * features[:, 0] - 0.1 * features[:, 1] + 1.2 * features[:, 2],
            ]
        )
        model, metrics = train_stiffness_surrogate(
            features,
            targets,
            hidden_sizes=(12, 8),
            epochs=2500,
            learning_rate=0.008,
            seed=5,
        )
        prediction = model.predict(features[:4])
        self.assertEqual(prediction.shape, (4, 3))
        self.assertLess(max(metrics["mape_percent"].values()), 8.0)

        with tempfile.TemporaryDirectory() as directory:
            model_file = save_stiffness_surrogate(model, Path(directory) / "model.npz")
            restored = load_stiffness_surrogate(model_file)
            np.testing.assert_allclose(restored.predict(features[:4]), prediction, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
