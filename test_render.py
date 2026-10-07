import time
from PIL import Image
from rcjmirror.config import SimParams
from rcjmirror.render import Simulator

p = SimParams()
t = time.time(); sim = Simulator(); print("init", round(time.time()-t, 1), "s")
sim.set_mirror()
for (x, y, h, name) in [(0, 0, 0, "a"), (-300, -500, 30, "b")]:
    p.robot.x, p.robot.y, p.robot.heading_deg = x, y, h
    t = time.time(); img = sim.render(p, 640, 480); print("render", round(time.time()-t, 2), "s")
    Image.fromarray(img).save(f"out_{name}.png")
X, Y, v, d = sim.floor_map(p, 21.0)
print("valid px", v.sum(), "dist range", d[v].min(), d[v].max())
