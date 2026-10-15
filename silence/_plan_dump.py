import json, random, sys
from silence import specs, build
packs_path, dev, out = sys.argv[1:4]
packs = build.load_packs(packs_path)
excl = set()
for l in open(dev):
    e = json.loads(l); excl.update(x for x in (e.get("pack"), e.get("pack2")) if x)
packs = {k: v for k, v in packs.items() if k not in excl}
specs.MUTTER_P = 0.7
s = build.PackSampler(packs, "train", 0)
r = random.Random("types:0:train")
types = r.choices(list(specs.TRAIN_MIX_V2), weights=list(specs.TRAIN_MIX_V2.values()), k=5600)
with open(out, "w") as f:
    for i, t in enumerate(types):
        p = s.next(build.family_of(t)); kw = {"pack2": s.next("B1")} if t == "B1_double" else {}
        f.write(json.dumps(specs.build(f"tr_{i:06d}", t, p, "train", "multi", 0, **kw), sort_keys=True) + "\n")
