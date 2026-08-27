# ComfyUI workflows

The LTX CLI cannot reproduce these. `ltx-pipelines`' argparse exposes no sampler,
scheduler, shift, sigma or denoise control — grepped for each, zero hits — and
hardcodes a sampler per pipeline (`LTX_2_3_HQ`'s own comment says "Res2s
sampler"). The community workflows set `euler_cfg_pp` /
`euler_ancestral_cfg_pp`, `linear_quadratic` scheduling, `LTXVScheduler` shift
2.05 / base 0.95 / terminal 0.1, and explicit `ManualSigmas` per pass. That is
where their quality comes from, and none of it is reachable from the CLI.

Source: RuneXX/LTX-2.3-Workflows on HuggingFace.
`..._Basic_for_checkpoint_models.json` is the one that matches our weights — it
uses `CheckpointLoaderSimple` against a monolith, which is what
`ltx-2.3-22b-dev.safetensors` is. The other variants want GGUF quantised files
we do not have.

## ui2api.py

Converts a UI-format workflow to the API format `/prompt` accepts. The editor
normally does this in the browser, which is why a downloaded workflow JSON cannot
simply be POSTed.

The awkward part is the **virtual nodes**. `Reroute`, `PrimitiveNode` and
KJNodes' `SetNode`/`GetNode` exist only in the editor and are not registered
server-side — confirmed against `/object_info`, where all four are absent while
every real node is present. They have to be resolved away: follow a `Reroute` to
its own upstream, inline a `PrimitiveNode`'s literal, and match a `GetNode` to
whatever fed the `SetNode` of the same name. Bypassed (`mode 4`) and muted
(`mode 2`) nodes are walked through the same way, since the API has no concept of
either flag.

Two more things the browser does that this has to replicate:

- **Widget values are positional.** Newer graphs list widget-backed inputs in the
  node's `inputs` array; older ones only carry `widgets_values` in the order the
  node's schema declares. The server's own `/object_info` is the only reliable
  source for which inputs are widgets rather than links. Seeds carry a hidden
  `control_after_generate` companion value that has to be skipped.
- **Trailing widgets are omitted.** The UI drops widgets it considers untouched,
  but the API format has no "unset" for a required input, so anything missing is
  filled from the schema default.

`REMAP` rewrites the author's own paths — including Windows separators, and
`-fp8` where we hold bf16.

## Usage

```bash
curl -s http://3090.zero:8191/object_info > objinfo.json
python3 ui2api.py <workflow>.json out.json objinfo.json
```

Passing `objinfo.json` is what enables schema-default filling and dropping nodes
the server does not know (shared workflows carry API/subgraph nodes that are
usually preview branches).

Three more traps, each of which produced a silently wrong graph rather than an
error, and each found only by submitting and reading ComfyUI's validator:

- **Widget order comes from the schema, never the node's `inputs` array.** That
  array lists only widgets the author had converted into inputs — node 165 shows
  just `width`/`height` while `widgets_values` carries all eight — so trusting it
  truncates the list and leaves every later widget on its default.
- **`widgets_values` is sometimes a dict**, keyed by input name (VHS_VideoCombine
  does this). `list()` on it yields the keys, which is how `loop_count` ended up
  holding the string `"loop_count"`.
- **Dynamic combos consume their sub-inputs inline**, addressed with a dotted
  name (`resize_type.multiplier`), matching the convention autogrow groups use
  (`variables.a`). Skipping them shifts every later widget by one.

## Known remaining gaps

- `LoadImage` points at the author's sample file and must be repointed.
- The sampler-preview VAE (`taeltx2_3`) is not present; `pixel_space` is the
  built-in fallback and affects only previews, not output.
- `SimpleCalculatorKJ`'s autogrow `variables` group is not fully modelled.
