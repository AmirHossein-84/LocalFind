# Test fixtures (P0-T003)

Committed, synthetic, freely redistributable samples (`synthetic-CC0`).
Tiny by design (~8 KB total) so unit/integration tests have real files
without bloating the repo. The full 120-asset corpus is generated on demand
into git-ignored `benchmarks/corpus/` — see `benchmarks/README.md`.

## Contents (`samples/`)

| File | What it is |
|---|---|
| `img_red.png` | Solid red frame, 320x240 |
| `img_bars.png` | SMPTE color bars, 320x240 |
| `vid_tones.mp4` | Two-tone clip: red 0–2000 ms, blue 2000–4000 ms (H.264, 30 fps) |

Referenced by `benchmarks/corpus.example.json` / `queries.example.json`
(corpus root for the example = this `samples/` dir).

## Regeneration recipe

```powershell
ffmpeg -hide_banner -loglevel error -y -f lavfi -i 'color=c=0xff0000:size=320x240:rate=30:duration=1' -frames:v 1 tests/fixtures/samples/img_red.png
ffmpeg -hide_banner -loglevel error -y -f lavfi -i 'smptebars=size=320x240:rate=30:duration=1' -frames:v 1 tests/fixtures/samples/img_bars.png
# vid_tones.mp4: two 2 s solid segments concatenated (see P0-T003 report)
```

## Rules (R-24)

- Only synthetic, consented, or redistributable files here. Never real user
  media, model weights, embeddings, secrets, or machine-specific paths.
- Private local fixtures belong in `tests/fixtures/private/` (git-ignored),
  never in this directory.
