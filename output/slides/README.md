# Management Agent classifier showcase

Open `Management_Agent_Classifier_Showcase.pptx` for editable slide text and the
native evaluation table. The app screenshots remain images. Open the PDF for a
portable visual copy, or `index.html` for a local viewer with keyboard navigation.

The eight slides cover resident intake, this project's classifier workflow,
classification results, human review, batch uploads, local evaluation and startup.
The PNG previews are 1920 by 1080 pixels. `contact-sheet.png` shows all slides.

This deck adapts the existing video showcase and includes the local Gemma results
measured on 3 October 2026. The original video demonstrates the offline baseline
and predates the live Ollama evaluation. The deck preserves that distinction.

To run the local prototype, open the downloaded project folder, install dependencies
with `uv sync --locked`, and run `start_project.cmd`. It starts or reuses Ollama and the website at
`http://localhost:8501/`. Select **Ollama** in the sidebar to use `gemma3:4b`.
For the offline classifier, run `start_project.cmd -SkipOllama`.

The PDF contains the rendered slide images. Use the PowerPoint file when you need
to edit the content. The deck has not been opened in native PowerPoint.

The final startup slide records the original demonstration workstation. Use the
portable instructions in the root README on another machine. Rebuilding the deck
with `scripts/build_showcase_slides.mjs` is optional and requires the Codex artifact
runtime and presentation skill; set `CODEX_MEDIA_RUNTIME` and
`CODEX_PRESENTATIONS_SKILL` to their local installation directories. These are not
dependencies of the classifier, PDF, or PowerPoint files.
