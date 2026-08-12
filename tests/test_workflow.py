import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import workflow


ROOT = Path(__file__).resolve().parents[1]


class WorkflowRegressionTests(unittest.TestCase):
    def make_direction(self, root: Path) -> Path:
        direction = root / "direction"
        direction.mkdir()
        (direction / "PLAN.md").write_text(
            """# Plan

## Success criterion (definition of \"done\")
Answer the feedback.

## Current status
Current status only.

## Next step
Do the requested plot.
""",
            encoding="utf-8",
        )
        (direction / "JOURNAL.md").write_text(
            """# Journal

## Old entry
OLD_JOURNAL_SENTINEL

## Latest entry
LATEST_JOURNAL_ONLY
""",
            encoding="utf-8",
        )
        (direction / "CHANGELOG.md").write_text(
            "CHANGELOG_SENTINEL " * 5000, encoding="utf-8"
        )
        return direction

    def prepare(self, direction: Path, feedback: str) -> Path:
        source = direction / "human_feedback_1.txt"
        source.write_text(feedback, encoding="utf-8")
        return workflow.create_manifest(direction, source)

    def route_and_seal(self, manifest: Path, outputs: list[str]) -> dict:
        data = workflow.load_manifest(manifest)
        data.update(
            state="ready",
            required_outputs=outputs,
            may_modify=outputs,
            must_remain_unchanged=[],
            unresolved_ambiguities=[],
        )
        manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        data = workflow.validate(manifest)
        data["baseline_sha256"] = workflow.snapshot_files(workflow.direction_for(manifest), data)
        data["state"] = "in_progress"
        manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return data

    def complete(self, manifest: Path) -> dict:
        data = workflow.load_manifest(manifest)
        for item in data["checklist"]:
            item["status"] = "done"
            item["completion"] = {
                "what_was_done": "Added the requested direct example.",
                "evidence": "REPORT_followup.md, Results.",
                "ambiguity": "None",
            }
        data["state"] = "review_pending"
        manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return workflow.validate(manifest, require_review=True)

    def test_verbatim_ampersand_and_compact_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            direction = self.make_direction(Path(tmp))
            feedback = (
                "2. Add example paths interpolating from & to trained characters.\n\n"
                "Write them into REPORT_followup.md."
            )
            manifest = self.prepare(direction, feedback)
            data = workflow.load_manifest(manifest)
            self.assertEqual(data["checklist"][0]["request"],
                             "2. Add example paths interpolating from & to trained characters.")
            self.assertIn("&", data["checklist"][0]["request"])
            self.assertNotIn("from and to", data["checklist"][0]["request"])

            context = workflow.compact_context(direction)
            self.assertIn("LATEST_JOURNAL_ONLY", context)
            self.assertNotIn("OLD_JOURNAL_SENTINEL", context)
            self.assertNotIn("CHANGELOG_SENTINEL", context)
            self.assertIn("REPORT_followup.md", context)

    def test_followup_is_formatted_but_content_review_can_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            direction = self.make_direction(Path(tmp))
            manifest = self.prepare(
                direction,
                "Write the answer into REPORT_followup.md.\nUse & literally.",
            )
            self.route_and_seal(manifest, ["REPORT_followup.md", "table.txt"])
            report = direction / "REPORT_followup.md"
            report.write_text(
                "# Follow-up\n\n## Research question\nA direct question.\n\n## Results\nA readable result.\n",
                encoding="utf-8",
            )
            (direction / "table.txt").write_text("requested output\n", encoding="utf-8")
            self.complete(manifest)

            review_files = workflow.files_for_review(manifest, workflow.load_manifest(manifest))
            self.assertIn("REPORT_followup.md", review_files)
            self.assertIn("table.txt", review_files)
            format_result = subprocess.run(
                ["python3", str(ROOT / "check_md.py"), str(report)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(format_result.returncode, 0, format_result.stdout + format_result.stderr)

            review_path = direction / ".tasks" / "review.json"
            review_path.write_text(
                json.dumps({
                    "pass": False,
                    "inspected_outputs": review_files,
                    "failures": ["The requested literal-& example is omitted."],
                    "summary": "Formatting passes, content does not.",
                }),
                encoding="utf-8",
            )
            self.assertFalse(workflow.record_review(manifest, review_path, format_passed=True))
            self.assertTrue((direction / "human_feedback_1.txt").exists())
            self.assertFalse((direction / "human_feedback_1.txt.addressed.md").exists())
            self.assertEqual(workflow.load_manifest(manifest)["state"], "rejected")

    def test_writing_file_is_short_and_passed_as_system_file(self):
        writing = (ROOT / "WRITING.md").read_text(encoding="utf-8")
        run = (ROOT / "run.sh").read_text(encoding="utf-8")
        self.assertLessEqual(len(writing.splitlines()), 30)
        self.assertIn("every `REPORT*.md`", writing)
        self.assertIn('--append-system-prompt-file "$WRITING_ABS"', run)
        self.assertNotIn("JOURNAL.md, RESULTS.md, and CHANGELOG.md in full", run)


if __name__ == "__main__":
    unittest.main()
