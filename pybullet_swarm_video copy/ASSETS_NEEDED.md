# 3D Assets Needed

Assets are prioritized by impact on perception module accuracy. The detector sees objects from ~10-15m altitude so silhouette shape, upright aspect ratio, and shadow footprint matter more than mesh complexity.

Good free sources: [Sketchfab](https://sketchfab.com), [Poly Haven](https://polyhaven.com), [cgtrader](https://www.cgtrader.com/free-3d-models).

---

## Targets -- what the detector needs to learn

| Asset | Notes |
|---|---|
| Soldier standing | Better than current rigged mesh -- needs clean human silhouette from top-down |
| Soldier crouching / prone | Different aspect ratio from above, harder detection case |
| Soldier in cover | Leaning against wall -- partial occlusion case |

---

## Vehicles -- scene context and occlusion sources

| Asset | Notes |
|---|---|
| Military light utility vehicle (HMMWV-style) | Troops move around these; large occluder |
| Destroyed / abandoned vehicle wreck | Static; replaces current box primitive |
| APC or IFV | Larger occluder; troops dismounting nearby |

---

## Structures -- replaces current box primitives

| Asset | Notes |
|---|---|
| Concrete jersey barrier / blast wall | Replaces current wall boxes |
| Ruined building section (partial walls, no roof) | Replaces `concrete_visual` box primitive |
| Guard post / watchtower base | Vertical structure, casts long shadow |
| Canvas tent / shelter | Soft cover; different texture signature from above |

---

## Ground clutter -- false positive challenge

| Asset | Notes |
|---|---|
| Stacked ammo crates | Roughly human-height stack; similar shadow footprint |
| Oil / fuel drum cluster | Cylindrical, upright, human-scale height |
| Sandbag wall section | Have single sandbag already -- need multi-bag wall config |
| Dead / stripped tree trunk | Vertical, casts shadow similar to a standing person |

---

## Terrain

| Asset | Notes |
|---|---|
| Rock formation / boulder cluster | Replaces current box rocks -- needs irregular silhouette |
| Crater with raised earth lip | Replaces current flat blast cylinder |
