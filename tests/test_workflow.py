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
        self.assertTrue((ROOT / "check_render.py").is_file())
        self.assertTrue((ROOT / "katex_compile.js").is_file())
        self.assertIn('python3 "$CHECK_RENDER_ABS"', run)
        self.assertNotIn("experiments/check_render.py", run)
        self.assertEqual(list(ROOT.glob("dir*/experiments/check_render.py")), [])
        self.assertEqual(list(ROOT.glob("dir*/experiments/katex_compile.js")), [])


    def test_default_word_budget_rejects_5001_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            direction = self.make_direction(Path(tmp))
            report = direction / "REPORT.md"
            report.write_text("word " * 5001, encoding="utf-8")
            self.assertIn("5001 words", workflow.report_budget_failures(direction)[0])

    def test_default_figure_budget_rejects_nine_embeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            direction = self.make_direction(Path(tmp))
            report = direction / "REPORT.md"
            report.write_text("\n".join(f"![figure {i}](plots/{i}.png)" for i in range(9)), encoding="utf-8")
            self.assertIn("9 Markdown image embeds", workflow.report_budget_failures(direction)[0])

    def test_task_override_permits_long_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            direction = self.make_direction(Path(tmp))
            report = direction / "REPORT.md"
            report.write_text("word " * 5001, encoding="utf-8")
            manifest = self.prepare(direction, "Write REPORT.md.")
            data = workflow.load_manifest(manifest)
            data["report_policy"]["max_words"] = 6000
            self.assertEqual(workflow.report_budget_failures(direction, ["REPORT.md"], data), [])

    def test_results_plots_need_not_be_in_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            direction = self.make_direction(Path(tmp))
            (direction / "REPORT.md").write_text("# Concise report\n\nNo main figure is needed.\n", encoding="utf-8")
            (direction / "RESULTS.md").write_text(
                "\n".join(f"![detail {i}](plots/{i}.png)" for i in range(20)), encoding="utf-8")
            self.assertEqual(workflow.report_budget_failures(direction), [])

    def make_feedback(self, direction: Path, number: int) -> Path:
        source = direction / f"human_feedback_{number}.txt"
        source.write_text(f"Feedback {number}.", encoding="utf-8")
        return workflow.create_manifest(direction, source)

    def approve_manifest(self, manifest: Path) -> bool:
        self.route_and_seal(manifest, ["REPORT.md"])
        direction = workflow.direction_for(manifest)
        (direction / "REPORT.md").write_text("# Report\n\nAnswer.\n", encoding="utf-8")
        self.complete(manifest)
        outputs = workflow.files_for_review(manifest, workflow.load_manifest(manifest))
        review = direction / ".tasks" / "approval.json"
        review.write_text(json.dumps({
            "pass": True, "inspected_outputs": outputs, "failures": [], "summary": "Approved."
        }), encoding="utf-8")
        return workflow.record_review(manifest, review, format_passed=True)

    def test_stop_with_feedback_enters_feedback_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            direction = self.make_direction(Path(tmp))
            (direction / "STOP").touch()
            self.make_feedback(direction, 1)
            self.assertEqual(workflow.begin_relaunch(direction), "feedback-only")
            self.assertTrue(workflow.feedback_only_marker(direction).is_file())
            self.assertFalse((direction / "STOP").exists())

    def test_rejection_or_remaining_feedback_does_not_restore_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            direction = self.make_direction(Path(tmp))
            (direction / "STOP").touch()
            first = self.make_feedback(direction, 1)
            self.make_feedback(direction, 2)
            workflow.begin_relaunch(direction)
            self.assertTrue(self.approve_manifest(first))
            self.assertFalse(workflow.restore_feedback_only_if_complete(direction))
            self.assertFalse((direction / "STOP").exists())
            active = workflow.active_manifests(direction)[0]
            self.route_and_seal(active, ["REPORT.md"])
            (direction / "REPORT.md").write_text("# Report\n", encoding="utf-8")
            self.complete(active)
            outputs = workflow.files_for_review(active, workflow.load_manifest(active))
            review = direction / ".tasks" / "rejection.json"
            review.write_text(json.dumps({"pass": False, "inspected_outputs": outputs,
                                          "failures": ["Incomplete"], "summary": "Rejected"}), encoding="utf-8")
            self.assertFalse(workflow.record_review(active, review, format_passed=True))
            self.assertFalse(workflow.restore_feedback_only_if_complete(direction))
            self.assertTrue(workflow.feedback_only_marker(direction).is_file())

    def test_final_approval_and_crash_recovery_restore_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            direction = self.make_direction(Path(tmp))
            (direction / "STOP").touch()
            manifest = self.make_feedback(direction, 1)
            workflow.begin_relaunch(direction)
            self.assertTrue(self.approve_manifest(manifest))
            self.assertTrue(workflow.restore_feedback_only_if_complete(direction))
            self.assertTrue((direction / "STOP").is_file())
            self.assertFalse(workflow.feedback_only_marker(direction).exists())
            (direction / "STOP").unlink()
            workflow.feedback_only_marker(direction).write_text("pending\n", encoding="utf-8")
            self.assertTrue(workflow.restore_feedback_only_if_complete(direction))
            self.assertTrue((direction / "STOP").is_file())

    def test_continue_research_bypasses_feedback_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            direction = self.make_direction(Path(tmp))
            (direction / "STOP").touch()
            self.make_feedback(direction, 1)
            self.assertEqual(workflow.begin_relaunch(direction, continue_research=True), "continue-research")
            self.assertFalse(workflow.feedback_only_marker(direction).exists())
            self.assertFalse((direction / "STOP").exists())


if __name__ == "__main__":
    unittest.main()
