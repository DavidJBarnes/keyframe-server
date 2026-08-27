#!/usr/bin/env python3
"""Convert a ComfyUI UI-format workflow to the API format /prompt accepts.

The UI format is a graph of nodes and link records meant for the editor. The API
format is a flat dict of {node_id: {class_type, inputs}} where every input is
either a literal or a [source_id, output_index] pair. The editor does the
conversion in the browser, which is why a workflow JSON cannot simply be POSTed.

The interesting part is the virtual nodes. Reroute, PrimitiveNode, and KJNodes'
SetNode/GetNode exist only in the editor -- they are NOT registered server-side
(confirmed against /object_info: all four are absent while every real node is
present). They have to be resolved away:

  Reroute      pass-through; follow to its own upstream
  PrimitiveNode drives a widget on its target rather than being an input
  SetNode      names a value; GetNode elsewhere reads that name
  GetNode      resolve to whatever the matching SetNode was fed

Bypassed (mode 4) and muted (mode 2) nodes also have to be walked through or
dropped, since the editor honours those flags and the API has no concept of them.
"""
import json, sys
from collections import defaultdict

# The author's paths, rewritten to what this box actually has. fp8 -> bf16
# because that is the checkpoint on disk; the spatial upscaler is 1.1 here.
REMAP = {
    "ltx-2.3-22b-dev-fp8.safetensors": "ltx-2.3-22b-dev.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    "gemma_3_12B_it_fp8_scaled.safetensors": "gemma_3_12B_it_fp8_scaled.safetensors",
}

VIRTUAL = {"Reroute", "PrimitiveNode", "SetNode", "GetNode",
           "Note", "MarkdownNote", "Fast Groups Bypasser (rgthree)",
           "PreviewAny", "VisualizeSigmasKJ", "easy showAnything"}


def load(path):
    d = json.load(open(path))
    nodes = {n["id"]: n for n in d["nodes"]}
    # link id -> (src_node, src_slot, dst_node, dst_slot)
    links = {l[0]: (l[1], l[2], l[3], l[4]) for l in d.get("links", []) if l}
    return d, nodes, links


def input_link(nodes, links, nid, name):
    """The link feeding input `name` of node `nid`, or None."""
    for inp in nodes[nid].get("inputs", []) or []:
        if inp.get("name") == name and inp.get("link") is not None:
            return links.get(inp["link"])
    return None


def resolve(nodes, links, setters, nid, slot, seen=None):
    """Walk back through virtual and bypassed nodes to a real (node, slot)."""
    seen = seen or set()
    if nid in seen:
        return None
    seen.add(nid)
    n = nodes.get(nid)
    if n is None:
        return None
    t = n.get("type")

    if t == "PrimitiveNode":
        # Drives a widget on its target rather than being a real input. Return
        # the literal it carries so the caller can inline it.
        wv = n.get("widgets_values") or [None]
        return ("__literal__", wv[0])

    if t == "Reroute":
        up = input_link(nodes, links, nid, "") or (
            links.get((n.get("inputs") or [{}])[0].get("link")) if n.get("inputs") else None)
        return resolve(nodes, links, setters, up[0], up[1], seen) if up else None

    if t == "GetNode":
        name = (n.get("widgets_values") or [None])[0]
        src = setters.get(name)
        if src is None:
            return None
        up = input_link(nodes, links, src, (nodes[src].get("inputs") or [{}])[0].get("name", ""))
        return resolve(nodes, links, setters, up[0], up[1], seen) if up else None

    if t == "SetNode":
        up = input_link(nodes, links, nid, (n.get("inputs") or [{}])[0].get("name", ""))
        return resolve(nodes, links, setters, up[0], up[1], seen) if up else None

    # Bypassed (4) / muted (2): pass the matching input through if one exists.
    if n.get("mode") in (2, 4):
        ins = n.get("inputs") or []
        if slot < len(ins) and ins[slot].get("link") is not None:
            up = links[ins[slot]["link"]]
            return resolve(nodes, links, setters, up[0], up[1], seen)
        for inp in ins:
            if inp.get("link") is not None:
                up = links[inp["link"]]
                return resolve(nodes, links, setters, up[0], up[1], seen)
        return None

    return (nid, slot)


def convert(path, registered=None, remap=None):
    """registered: node types the server knows. Anything else is dropped, along
    with whatever only fed it -- API/subgraph nodes appear in shared workflows
    and are usually preview branches rather than load-bearing.

    remap: {substring: replacement} applied to string inputs. Shared workflows
    carry the author's own paths, including Windows separators, which mean
    nothing here."""
    d, nodes, links = load(path)
    setters = {}
    for nid, n in nodes.items():
        if n.get("type") == "SetNode":
            nm = (n.get("widgets_values") or [None])[0]
            if nm:
                setters[nm] = nid

    api = {}
    for nid, n in nodes.items():
        t = n.get("type")
        if t in VIRTUAL or n.get("mode") in (2, 4) or t is None:
            continue
        if registered is not None and t not in registered:
            print(f"  dropping unregistered node {nid} ({t})")
            continue
        entry = {"class_type": t, "inputs": {}, "_meta": {"title": n.get("title") or t}}
        # Widget values, positionally.
        #
        # Two UI formats exist. Newer graphs list widget-backed inputs in the
        # node's `inputs` array with a "widget" key; older ones only carry
        # `widgets_values` and rely on the order the node's schema declares. Fall
        # back to the server's own schema for the second case, which is also the
        # only source of truth for which inputs are widgets rather than links.
        wv = list(n.get("widgets_values") or [])
        widget_names = [i.get("name") for i in (n.get("inputs") or [])
                        if i.get("widget") is not None]
        if not widget_names and registered is not None:
            spec = (registered.get(t) or {}).get("input", {}).get("required", {})
            for name, meta in spec.items():
                typ = meta[0]
                # A list of options, or a primitive, is a widget; anything else
                # (MODEL, CLIP, LATENT, IMAGE ...) arrives over a link.
                if isinstance(typ, list) or typ in ("INT", "FLOAT", "STRING",
                                                    "BOOLEAN", "COMBO"):
                    widget_names.append(name)
        wi = 0
        for name in widget_names:
            if wi >= len(wv):
                break
            entry["inputs"][name] = wv[wi]
            wi += 1
            # Seeds carry a hidden "control_after_generate" companion value in
            # the UI that has no counterpart in the API schema.
            if name in ("seed", "noise_seed") and wi < len(wv) and \
                    isinstance(wv[wi], str) and wv[wi] in (
                        "fixed", "increment", "decrement", "randomize"):
                wi += 1
        # linked inputs override widgets
        for inp in n.get("inputs") or []:
            if inp.get("link") is None:
                continue
            up = links.get(inp["link"])
            if not up:
                continue
            r = resolve(nodes, links, setters, up[0], up[1])
            if r and r[0] == "__literal__":
                entry["inputs"][inp["name"]] = r[1]
            elif r:
                entry["inputs"][inp["name"]] = [str(r[0]), r[1]]
        # Anything still unset gets the schema's own default. The UI omits
        # trailing widgets it considers untouched, but the API format has no
        # notion of "unset" for a required input -- it must be present.
        if registered is not None:
            spec = (registered.get(t) or {}).get("input", {}).get("required", {})
            for name, meta in spec.items():
                if name in entry["inputs"]:
                    continue
                typ = meta[0]
                opts = meta[1] if len(meta) > 1 and isinstance(meta[1], dict) else {}
                if "default" in opts:
                    entry["inputs"][name] = opts["default"]
                elif isinstance(typ, list) and typ:
                    entry["inputs"][name] = typ[0]
                elif typ == "COMBO" and opts.get("options"):
                    entry["inputs"][name] = opts["options"][0]

        if remap:
            for k, v in list(entry["inputs"].items()):
                if isinstance(v, str):
                    for frm, to in remap.items():
                        if frm in v:
                            entry["inputs"][k] = to
                            print(f"  remap {nid}.{k}: {v!r} -> {to!r}")
                            break
        api[str(nid)] = entry

    # Drop links to nodes that no longer exist (e.g. the dropped preview branch).
    ids = set(api)
    for nid, n in api.items():
        for k, v in list(n["inputs"].items()):
            if isinstance(v, list) and len(v) == 2 and str(v[0]) not in ids:
                print(f"  pruning dead link {nid}.{k} -> {v[0]}")
                del n["inputs"][k]
    return api


if __name__ == "__main__":
    reg = None
    if len(sys.argv) > 3:
        reg = json.load(open(sys.argv[3]))
    api = convert(sys.argv[1], registered=reg, remap=REMAP)
    out = sys.argv[2] if len(sys.argv) > 2 else "api.json"
    json.dump(api, open(out, "w"), indent=1)
    print(f"{len(api)} nodes -> {out}")
    unresolved = sum(1 for v in api.values() for x in v["inputs"].values()
                     if x is None)
    print(f"unresolved inputs: {unresolved}")
