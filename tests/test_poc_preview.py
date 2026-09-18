import re
import unittest

from app.main import UI_HTML


class PocPreviewTests(unittest.TestCase):
    def _function_body(self, name: str, next_name: str) -> str:
        match = re.search(
            rf"(?:async )?function {name}\([^)]*\) \{{(?P<body>.*?)\n    (?:async )?function {next_name}\(",
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

    def test_mesh_uses_reference_while_shape_pca_stays_parametric(self):
        exact_body = self._function_body("exactUploadedGeometryContext", "runGmshMesh")
        mesh_body = self._function_body("runGmshMesh", "runShapePca")
        pca_body = self._function_body("runShapePca", "cleanupMeshViewer")

        self.assertNotIn("pocRequirementsComplete", exact_body)
        self.assertIn("meshRequestOptions(intent)", mesh_body)
        self.assertIn("meshRequestOptions(intent, false)", pca_body)
        self.assertIn('mesh_strategy: "uploaded_geometry_all_hex"', UI_HTML)

    def test_requirement_generate_is_preview_only(self):
        requirements_body = self._function_body("pocRequirementsHtml", "bindPocRequirements")
        generate_body = self._function_body("generateFourArmPocCad", "generateDesignSpaceVariants")
        upload_body = self._function_body("uploadContextFile", "buildFullPrompt")

        self.assertNotIn("Client conditions", requirements_body)
        self.assertIn("Inner-core diameter (mm)", requirements_body)
        self.assertIn("Inner-core length (mm)", requirements_body)
        self.assertIn("Outer-core length (mm)", requirements_body)
        self.assertIn("generateRequirementBtn", requirements_body)
        self.assertNotIn("activateParametricDesignWorkflow", generate_body)
        self.assertNotIn("buildRubberBushingWorkflow", generate_body)
        self.assertIn("setParamEditorOpen(false)", generate_body)
        self.assertIn("setWorkflowToolsVisible(false)", generate_body)
        self.assertIn("activateParametricDesignWorkflow", upload_body)
        self.assertIn("buildParamControls(currentEditIntent)", upload_body)


if __name__ == "__main__":
    unittest.main()
