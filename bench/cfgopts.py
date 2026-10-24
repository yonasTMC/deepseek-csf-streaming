import dataclasses, inspect, re
from b12x.moe.fused_moe import _tuning as t
print([(f.name, f.default) for f in dataclasses.fields(t.MoeDecodeConfig)])
src = inspect.getsource(t)
for key in ("backend", "dynamic_route_mode", "route_planner"):
    print(key, sorted(set(re.findall(key + r"\s*(?:==|!=|in)\s*\(?[\"']([a-z_0-9]+)", src))))
print("backend= literals", sorted(set(re.findall(r"backend=[\"']([a-z_]+)", src))))
print("dynamic_tile_m literals", sorted(set(re.findall(r"dynamic_tile_m=(\d+)", src))))
