# Flick Canceling SDÑ

> Gambiarra is part of creativity, too.
> *(Gambiarra: Brazilian for a clever, improvised fix that shouldn't work, but does.)*

**English** · [Português](README.pt-BR.md)

Made by [Smart Diseños (SDÑ)](https://smartdisenos.com), a *Non-Traditional Creative Studio*, in the middle of a real 3D film, when the water and ice in the render started to flicker and there was no time to render again. We're sharing it for anyone stuck in the same spot.

Think of it like noise canceling, but for flicker. It is a studio tool that reduces light and color blotches that pop in and out of a shot. It outputs four signed RGB deltas, their sum, and a PNG mask of the pixels any layer touched. How much of it you use is decided in comp.

## Install and use

Use Python 3.14 in a virtual environment. Number your frames with no gaps, keeping the same resolution and color curve across the shot. The recipe was designed for RGB on a 0 to 1 scale. PNG and TIFF accept 8 or 16 bits. EXR uses the main R, G and B channels. The tool works on the values it receives, with no color transform.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/flick run --entrada 'frames/light_%04d.png' --inicio 1 --fim 48 \
  --regiao 'masks/object_%04d.png' --excluir masks/protect.png \
  --receita flick/receita.json --processos 4 --saida out/shot01
.venv/bin/flick decide --entrada 'frames/light_%04d.png' --inicio 1 --fim 48 \
  --execucao out/shot01 --saida out/decide01
.venv/bin/flick apply --entrada 'frames/light_%04d.png' --inicio 1 --fim 48 \
  --delta 'out/decide01/escolha/delta_%04d.exr' --forca 1 --saida out/preview01
.venv/bin/flick measure --entrada 'frames/light_%04d.png' --inicio 1 --fim 48 \
  --corrigida 'out/preview01/quadro_%04d.png' --regiao 'masks/object_%04d.png' \
  --trecho 'mid=10,38,2' --saida out/measure01
.venv/bin/python -m unittest discover -s tests
```

The flags are in Portuguese, because that's how the studio thinks: `entrada` input, `inicio`/`fim` first/last frame, `regiao` region, `excluir` exclude, `receita` recipe, `processos` processes, `saida` output, `forca` strength, `corrigida` corrected, `trecho` range, `execucao` run folder.

## Let it decide

You shouldn't have to judge every shot by eye. `flick decide` reviews its own work: it applies the layers one by one (1, 1+2, 1+2+3, all four), measuring flicker and detail only where the tool acted. A combination is vetoed when it keeps less than 90% of the detail (`--detalhe-min`) or when more than 1% of the 64-pixel tiles gain too much detail, a sign of doubled edges (`--fantasma-max`). Among the rest, it keeps the smallest combination within 2 points of the best one (`--ganho-min`), so a layer that barely helps stays out. Layers are not monotonic: one can do almost nothing and the next one a lot, so all of them are measured before deciding. It writes the chosen delta to `escolha/` and explains the choice in `decisao.json`.

On a 3D render with flickering ice, it kept all four layers: 30% less flicker, 93% of the detail. On a public-domain timelapse ([Growing grass, flicker not removed](https://commons.wikimedia.org/wiki/File:Growing_grass_(flicker_not_removed).ogv)), it kept layers 1 to 3: 55% less flicker with all the detail, against 28% for a plain global-brightness deflicker.

Each delivery command asks for a brand-new folder. The preview is `clip(original + strength × delta, 0, 1)`. Strength zero with the same format copies the original. The delta EXR is signed and uses 16-bit half by default. Use `--precisao float32` to write the sum and the spatial reduction with more precision. Processing runs in float32 and keeps the half rounding of the reference recipe's intermediate writes.

## Recipe and masks

The first two layers stabilize small tiles with ECC affine registration, pick an appearance reference by luminance with memory, and protect consistent motion. The second shifts the grid over the result of the first. The third detects alternation against the average of the two neighbors. The fourth follows manual masks and automatic regions, smoothing the color trajectory inside each stabilized region.

- Defaults live in `flick/receita.json`. A partial JSON replaces only the fields it provides. On the CLI, for example, `--param camada2.sigma_cor=4.0` or `--param camada4.automatica=false`. Distances scale by screen width divided by 3840, areas by the square of that ratio. Temporal window sizes are in frames. Region dilation and smoothing use the mask's native width divided by 1920.
- `--regiao` takes a fixed or per-frame grayscale PNG. White allows the correction, black excludes it. The region envelope may grow past the edge as the recipe says. `--excluir` protects every marked pixel, manual layer included. The manual mask has its own envelope.
- For Cryptomatte, use `--cryptomatte 'passes/matte_%04d.exr' --camada-crypto CryptoObject --id-crypto Glass`. Repeat the flag to add coverages. It takes names from the embedded manifest or the 8-digit hex ID. The region uses the sum of weights of each ID/coverage pair.
- To follow an object, use `--mascara-track '{"arquivo":"mask.png","quadro":24}'`. It also takes JSON with `{"pontos":[[x1,y1],[x2,y2],[x3,y3]]}`. The optional `janela` key takes `[x,y,width,height]` to bound the registration. Coordinates start at the top left of the full screen. Paths inside recipes are relative to the JSON. The default automatic reference is the middle of the sequence.
- `--faixa X Y W H` processes a crop and writes zero outside it. `--tamanho-saida W H` sets the delivery resolution. The NPY adapter takes `[time,height,width,BGR]`, unsigned integers or normalized floats. For an already-cropped NPY, declare `--tela W H` and the crop. `--origem-npy` names the first stored frame when extracting a range.
- `--entregar Q0 Q1` limits the exported range. The layers use every frame from `--inicio` to `--fim` as context.
- `--trava-fluxo PX` adds protection against strong motion on export. Optical flow looks for the same point in both neighbor frames and checks the way back to the current frame. Weight falls linearly between half of PX and PX pixels of motion per frame, measured at input resolution. A pixel loses the correction when it leaves the region in a neighbor, fails the round trip or hits the speed limit. In each channel, only the part of the delta that moves toward the center of both neighbors in the same direction remains. The ends of the context sequence get zero delta. `trava_fluxo.json` reports coverage per frame. The original recipe stays the default.
- `--trava-textura 0.70`, together with the flow lock, compares luminance in 15 × 15 pixel windows in both aligned neighbors. A local correlation up to that threshold vetoes the action. Above it, weight grows linearly to 1. The contrast control also softens windows whose standard deviation stays below 0.008 in normalized code values.
- `--teto-delta 0.04`, also optional with the flow lock, gradually compresses amplitudes above half the ceiling. The largest channel shift approaches the ceiling and the three channels get the same factor. Each layer is quantized after this step.

## Measurements

`medidas.json` and `medidas.md` report the reductions in total jump, wide jump, red, swap, detail variation, and the largest local detail increase. Each center uses two previous and two following neighbors. The largest local increase covers the whole frame. `registro1.json` to `registro4.json` keep the registration's selection and correlation diagnostics. `execucao.json` logs times and, on macOS, sampled memory peaks of the process tree. Disk-mapped caches stay in the run folder.

## Prior art

We didn't invent localized flicker correction. Earlier work includes Pitié et al. (2006) and The Foundry's Furnace (2011). [CVMP paper](https://doi.org/10.1049/cp:20061935) · [Furnace 4.2 manual, p. 79](https://thefoundry.s3.amazonaws.com/products/furnace/nuke/documentation/Furnace_4.2_Nuke.pdf#page=79)

## License

Apache 2.0, see `LICENSE`. Hola from Brazil. 🇧🇷
