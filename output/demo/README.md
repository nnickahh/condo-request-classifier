# Classifier demonstration

`Management_Agent_Classifier_Demo.mp4` is a roughly 79-second, 1080p presentation
with English narration and burned-in captions. It is an edited walkthrough built
from screenshots captured from the working Streamlit app, with title cards and
crossfades; it is not an uncut screen recording or a latency benchmark.

The video covers resident intake, a maintenance classification, an emergency
review flag, a four-row CSV inbox and the synthetic evaluation dashboard.
All demonstrated requests are synthetic. The workflow includes deterministic
safety rules, and the video distinguishes the working local baseline from the
Ollama adapter, whose live inference had not yet been tested when this video was
made. Later local Gemma results are documented in the root README and slide deck.

## Files

- `Management_Agent_Classifier_Demo.mp4`: shareable H.264/AAC video.
- `captions.srt`: separate caption track for editors or presentation tools.
- `poster.png`: title-card image.
- `timeline.json`: scene times and narration text.
- `captures/`: original screenshots from the live app.
- `stills/`: composed scene images.
- `audio/` and `narration.wav`: locally synthesized English narration.

## Rebuild

Capture current app screens into the filenames listed in
`scripts/demo_storyboard.json` before rebuilding. Browser captures are 1280x720.
The script composes the video at 1920x1080 and does not control a browser.

On Windows, generate narration with the installed Microsoft Hazel Desktop voice:

```powershell
.\scripts\narrate_demo.ps1
uv run --with imageio-ffmpeg --with pillow --with numpy python scripts/render_demo.py
```

Narration is generated locally using Windows speech synthesis. No third-party
voice service, music recording or resident audio is used.
