// Compile LaTeX snippets with KaTeX and report errors.
// stdin: JSON list of {id, tex, display}. stdout: JSON list of {id, ok, error}.
let katex;
try {
  katex = require("katex");
} catch (e) {
  katex = require("/tmp/katexcheck/node_modules/katex");
}
const chunks = [];
process.stdin.on("data", (c) => chunks.push(c));
process.stdin.on("end", () => {
  const items = JSON.parse(chunks.join(""));
  const out = items.map((it) => {
    try {
      katex.renderToString(it.tex, {
        displayMode: it.display,
        throwOnError: true,
        strict: "error",
      });
      return { id: it.id, ok: true, error: "" };
    } catch (e) {
      return { id: it.id, ok: false, error: String(e.message || e) };
    }
  });
  process.stdout.write(JSON.stringify(out));
});
