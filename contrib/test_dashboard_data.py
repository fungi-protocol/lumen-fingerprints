import json, os, sys, tempfile, unittest, subprocess

class FieldsEmitTest(unittest.TestCase):
    def test_fields_emitted_as_dict_keyed_by_name(self):
        """Test that report.json's fields array is emitted as a dict keyed by field name."""
        # Build a minimal report.json-shaped dict with a fields array
        report = {
            "window": {"start_height": 100, "end_height": 200},
            "totals": {"tx_count": 1000},
            "axis_summaries": [],
            "encoding_families": {},
            "conditional_anonymity": {},
            "fields": [
                {
                    "name": "script_type",
                    "section": "input",
                    "kind": "categorical",
                    "total": 1000,
                    "values": {"p2pkh": 600, "p2sh": 400}
                },
                {
                    "name": "input_count",
                    "section": "transaction",
                    "kind": "numeric",
                    "total": 1000,
                    "stats": {"mean": 2.5, "median": 2},
                    "hist": [{"bin": "1", "count": 500}]
                }
            ]
        }

        # Write temp report.json
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(report, f)
            report_path = f.name

        # Write temp explorer-data.json path
        explorer_path = tempfile.mktemp(suffix=".json")

        try:
            # Run dashboard-data.py as subprocess
            script_path = os.path.join(os.path.dirname(__file__), "dashboard-data.py")
            result = subprocess.run(
                [sys.executable, script_path, report_path, explorer_path],
                capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")

            # Read and verify output
            with open(explorer_path) as f:
                explorer = json.load(f)

            # Assert fields key exists and is a dict
            self.assertIn("fields", explorer)
            self.assertIsInstance(explorer["fields"], dict)

            # Assert fields are keyed by name
            self.assertIn("script_type", explorer["fields"])
            self.assertIn("input_count", explorer["fields"])

            # Verify categorical field structure
            script_type_field = explorer["fields"]["script_type"]
            self.assertEqual(script_type_field["name"], "script_type")
            self.assertEqual(script_type_field["kind"], "categorical")
            self.assertEqual(script_type_field["values"], {"p2pkh": 600, "p2sh": 400})

            # Verify numeric field structure
            input_count_field = explorer["fields"]["input_count"]
            self.assertEqual(input_count_field["name"], "input_count")
            self.assertEqual(input_count_field["kind"], "numeric")
            self.assertEqual(input_count_field["stats"], {"mean": 2.5, "median": 2})
            self.assertEqual(input_count_field["hist"], [{"bin": "1", "count": 500}])

            # Verify existing keys are preserved
            self.assertIn("window", explorer)
            self.assertIn("totals", explorer)
            self.assertIn("axis_summaries", explorer)
            self.assertIn("encoding_families", explorer)
            self.assertIn("cond", explorer)

        finally:
            # Cleanup
            if os.path.exists(report_path):
                os.unlink(report_path)
            if os.path.exists(explorer_path):
                os.unlink(explorer_path)

    def test_fields_empty_when_missing_from_report(self):
        """Test that fields defaults to empty dict when not in report."""
        report = {
            "window": {"start_height": 100, "end_height": 200},
            "totals": {"tx_count": 1000},
            "axis_summaries": [],
            "encoding_families": {},
            "conditional_anonymity": {}
        }

        # Write temp report.json
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(report, f)
            report_path = f.name

        # Write temp explorer-data.json path
        explorer_path = tempfile.mktemp(suffix=".json")

        try:
            # Run dashboard-data.py as subprocess
            script_path = os.path.join(os.path.dirname(__file__), "dashboard-data.py")
            result = subprocess.run(
                [sys.executable, script_path, report_path, explorer_path],
                capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")

            # Read and verify output
            with open(explorer_path) as f:
                explorer = json.load(f)

            # Assert fields key exists and is an empty dict
            self.assertIn("fields", explorer)
            self.assertIsInstance(explorer["fields"], dict)
            self.assertEqual(explorer["fields"], {})

        finally:
            # Cleanup
            if os.path.exists(report_path):
                os.unlink(report_path)
            if os.path.exists(explorer_path):
                os.unlink(explorer_path)

if __name__ == "__main__":
    unittest.main()
