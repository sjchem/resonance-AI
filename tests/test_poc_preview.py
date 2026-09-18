import re
import unittest

from app.main import UI_HTML


class PocPreviewTests(unittest.TestCase):
    def _function_body(self, name: str, next_name: str) -> str:
        match = re.search(
            rf"function {name}\([^)]*\) \{{(?P<body>.*?)\n    function {next_name}\(",
            UI_HTML,
            re.DOTALL,
        )
        self.assertIsNotNone(match, f"Could not find JavaScript function {name}")
        return match.group("body")

    def test_design_selection_preserves_reference_preview(self):
        load_body = self._function_body("loadVariant", "applyRubberDesignIntent")
        best_body = self._function_body("findBestGeometry", "relativeErrorMagnitude")
        apply_body = self._function_body("applyRubberDesignIntent", "captureTargetSearchInputs")

        self.assertNotIn("generateRubberParametricCad", load_body)
        self.assertNotIn("generateRubberParametricCad", best_body)
        self.assertNotIn("warpEditableMeshFaces", apply_body)
        self.assertIn("pickUploadedMesh()", apply_body)
        self.assertIn("render3DPreview(currentEditIntent)", apply_body)


if __name__ == "__main__":
    unittest.main()
