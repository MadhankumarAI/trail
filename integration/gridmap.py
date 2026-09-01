"""ANYbotics grid_map interoperability for the 2.5D terrain layer.

WHY THIS EXISTS, AND WHY IT IS NOT THE INTERNAL STORE
-----------------------------------------------------
grid_map (github.com/ANYbotics/grid_map) is the standard 2.5D representation in
robotics: a rectangular grid of named float layers -- elevation, variance,
traversability -- over a circular buffer, with a ROS message, an RViz plugin
and a large ecosystem (elevation_mapping, traversability_estimation) already
speaking it. Emitting it is how this work reaches the rest of a ROS 2 stack
without anyone writing a bridge.

But grid_map is FIXED RESOLUTION. One `resolution` scalar covers the whole map,
by construction: the circular buffer and every iterator depend on it. The
problem statement asks for the opposite -- 5 cm cells within 10 m degrading to
50 cm at 100 m -- so grid_map cannot be the internal representation without
abandoning the requirement.

So it is used for what it is genuinely best at, in two roles:

  1. INTERCHANGE. The map is serialised to grid_map_msgs/GridMap so RViz and
     any existing grid_map consumer can read it directly.
  2. BASELINE. A dense uniform grid is exactly what the adaptive scheme claims
     to improve on, and grid_map is the honest, industry-standard version of
     that baseline -- far more convincing than a strawman we wrote ourselves.
     `compare_memory` measures one against the other.

LAYER NAMES
-----------
Taken from grid_map and traversability_estimation rather than invented, so the
output drops into existing tooling:

    elevation                   metres, in the map frame
    variance                    elevation variance (elevation_mapping)
    n_observations              how many returns support the cell
    traversability              0..1, 1 = freely traversable
    traversability_slope        per-criterion sublayers, as
    traversability_step         traversability_estimation publishes them
    traversability_roughness
    obstacle_height             metres a static obstacle stands above ground

The traversability convention is inverted from our cost, and the mapping is
chosen so the two thresholds coincide exactly:

    traversability = clip(1 - score/2, 0, 1)

Our score is the worst normalised violation, so score = 1 is exactly the
vehicle limit; that maps to traversability 0.5, which is precisely
traversability_estimation's default `traversability_threshold`. A consumer
using stock parameters therefore agrees with us about what is drivable,
without being told anything.

GEOMETRY, TAKEN FROM GridMapMath.cpp
------------------------------------
Index axis 0 is x, axis 1 is y, and index (0, 0) sits at MAXIMUM x and maximum
y -- indices grow towards decreasing coordinates:

    position = mapPosition + (0.5 * length - 0.5 * res) - res * index

Getting that backwards produces a map that looks entirely plausible and is
mirrored, which is the kind of bug that survives a demo and fails in the field.
`selftest()` checks the round trip against the corners.
"""

import numpy as np

LAYERS = (
    "elevation",
    "variance",
    "n_observations",
    "traversability",
    "traversability_slope",
    "traversability_step",
    "traversability_roughness",
    "obstacle_height",
)


def cost_to_traversability(score):
    """Our cost (0 good, 1 = vehicle limit) -> grid_map's 0..1, 1 = best.

    score 1.0 lands on 0.5, which is traversability_estimation's default
    threshold, so a stock consumer draws the same line we do.
    """
    return np.clip(1.0 - 0.5 * np.asarray(score, float), 0.0, 1.0)


class GridMap:
    """A grid_map-compatible multi-layer 2.5D map.

    Dense on purpose: this mirrors grid_map's own storage so the memory figure
    it reports is the real one a grid_map user would pay, not a flattering
    approximation.
    """

    def __init__(self, length_x, length_y, resolution, position=(0.0, 0.0),
                 frame_id="map", layers=LAYERS):
        self.resolution = float(resolution)
        # grid_map rounds the size up to whole cells and keeps length consistent
        self.n_x = int(round(length_x / self.resolution))
        self.n_y = int(round(length_y / self.resolution))
        self.length_x = self.n_x * self.resolution
        self.length_y = self.n_y * self.resolution
        self.position = np.array(position, float)
        self.frame_id = frame_id
        self.layer_names = list(layers)
        self.layers = {k: np.full((self.n_x, self.n_y), np.nan, np.float32)
                       for k in self.layer_names}
        # circular-buffer origin; a freshly built map starts unrotated
        self.start = np.zeros(2, int)

    # ------------------------------------------------------------ geometry
    def index_from_position(self, x, y):
        """(ix, iy) for world positions, and a mask of what is inside."""
        ox = self.position[0] + 0.5 * self.length_x - 0.5 * self.resolution
        oy = self.position[1] + 0.5 * self.length_y - 0.5 * self.resolution
        ix = np.rint((ox - np.asarray(x, float)) / self.resolution).astype(int)
        iy = np.rint((oy - np.asarray(y, float)) / self.resolution).astype(int)
        ok = (ix >= 0) & (ix < self.n_x) & (iy >= 0) & (iy < self.n_y)
        return ix, iy, ok

    def position_from_index(self, ix, iy):
        ox = self.position[0] + 0.5 * self.length_x - 0.5 * self.resolution
        oy = self.position[1] + 0.5 * self.length_y - 0.5 * self.resolution
        return ox - np.asarray(ix) * self.resolution, \
            oy - np.asarray(iy) * self.resolution

    def move(self, new_position):
        """Relocate the map without copying the data that stays in view.

        grid_map's defining trick: the overlap is preserved by rotating the
        circular buffer and clearing only the strips that just came into view.
        Implemented here with np.roll, which costs a copy numpy cannot avoid,
        but the SEMANTICS are the point -- cells keep their world identity when
        the robot moves, and only genuinely new ground is blanked.
        """
        new_position = np.array(new_position, float)
        shift = np.rint((new_position - self.position) /
                        self.resolution).astype(int)
        if not shift.any():
            return np.zeros(2, int)

        # indices grow towards decreasing position, hence the sign flip
        roll = (int(shift[0]), int(shift[1]))
        for k in self.layer_names:
            a = self.layers[k]
            a = np.roll(a, roll, axis=(0, 1))
            if roll[0] > 0:
                a[:roll[0], :] = np.nan
            elif roll[0] < 0:
                a[roll[0]:, :] = np.nan
            if roll[1] > 0:
                a[:, :roll[1]] = np.nan
            elif roll[1] < 0:
                a[:, roll[1]:] = np.nan
            self.layers[k] = a
        self.position = self.position + shift * self.resolution
        return shift

    # --------------------------------------------------------------- data
    def set_cells(self, x, y, values):
        """Scatter world-positioned samples into layers. `values` is a dict."""
        ix, iy, ok = self.index_from_position(x, y)
        ix, iy = ix[ok], iy[ok]
        for k, v in values.items():
            if k not in self.layers:
                raise KeyError(f"no layer {k!r}; have {self.layer_names}")
            self.layers[k][ix, iy] = np.asarray(v, np.float32)[ok]
        return int(ok.sum())

    def occupancy(self):
        e = self.layers["elevation"]
        return float(np.isfinite(e).mean())

    # ------------------------------------------------------------ message
    def to_msg(self, stamp_sec=0, stamp_nanosec=0):
        """A dict matching grid_map_msgs/GridMap exactly.

        Layer data is a Float32MultiArray in COLUMN-MAJOR order with dims
        [column_index = n_y, row_index = n_x] -- grid_map copies an Eigen
        matrix straight out, and Eigen is column-major by default. Writing it
        row-major yields a transposed map that still renders, which is why the
        order is stated here rather than left to the reader.
        """
        out_layers = []
        for k in self.layer_names:
            a = self.layers[k]
            out_layers.append({
                "layout": {
                    "dim": [
                        {"label": "column_index", "size": self.n_y,
                         "stride": self.n_x * self.n_y},
                        {"label": "row_index", "size": self.n_x,
                         "stride": self.n_x},
                    ],
                    "data_offset": 0,
                },
                # order="F" is the column-major flatten Eigen would produce
                "data": a.flatten(order="F").tolist(),
            })
        return {
            "info": {
                "header": {"stamp": {"sec": int(stamp_sec),
                                     "nanosec": int(stamp_nanosec)},
                           "frame_id": self.frame_id},
                "resolution": self.resolution,
                "length_x": self.length_x,
                "length_y": self.length_y,
                "pose": {
                    "position": {"x": float(self.position[0]),
                                 "y": float(self.position[1]), "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
            "layers": self.layer_names,
            "basic_layers": ["elevation"],
            "data": out_layers,
            "outer_start_index": int(self.start[1]),
            "inner_start_index": int(self.start[0]),
        }

    # ------------------------------------------------------------- memory
    def memory_bytes(self):
        """What grid_map actually holds: every cell of every layer, always."""
        return self.n_x * self.n_y * len(self.layer_names) * 4


def selftest():
    """Round-trip the geometry against the corners.

    A mirrored map is the failure mode here, and it looks fine.
    """
    g = GridMap(20.0, 10.0, 0.5, position=(3.0, -1.0))
    assert (g.n_x, g.n_y) == (40, 20), (g.n_x, g.n_y)

    # index (0,0) must sit at MAXIMUM x and y
    px, py = g.position_from_index(0, 0)
    assert abs(px - (3.0 + 10.0 - 0.25)) < 1e-9, px
    assert abs(py - (-1.0 + 5.0 - 0.25)) < 1e-9, py

    # and the far corner at minimum x and y
    px, py = g.position_from_index(g.n_x - 1, g.n_y - 1)
    assert abs(px - (3.0 - 10.0 + 0.25)) < 1e-9, px
    assert abs(py - (-1.0 - 5.0 + 0.25)) < 1e-9, py

    # round trip a scatter of positions
    rng = np.random.default_rng(0)
    x = rng.uniform(-6.0, 12.0, 500)
    y = rng.uniform(-5.0, 3.0, 500)
    ix, iy, ok = g.index_from_position(x, y)
    bx, by = g.position_from_index(ix[ok], iy[ok])
    assert np.abs(bx - x[ok]).max() <= g.resolution / 2 + 1e-9
    assert np.abs(by - y[ok]).max() <= g.resolution / 2 + 1e-9

    # the centre cell of an even-sized map straddles the position by half a
    # cell, which is grid_map's behaviour and not a rounding slip
    ix, iy, ok = g.index_from_position(np.array([3.0]), np.array([-1.0]))
    assert ok[0] and ix[0] in (19, 20) and iy[0] in (9, 10), (ix, iy)

    # move() must keep the overlap and blank only what is newly exposed
    g.layers["elevation"][:] = 1.0
    g.move((3.0 + 2.0, -1.0))          # 4 cells of new ground in x
    e = g.layers["elevation"]
    assert np.isnan(e[:4, :]).all(), "newly exposed strip not cleared"
    assert (e[4:, :] == 1.0).all(), "overlap was not preserved"

    # message shape and ordering
    m = GridMap(2.0, 1.0, 0.5).to_msg()
    assert m["layers"] == list(LAYERS)
    d = m["data"][0]
    assert d["layout"]["dim"][0]["size"] == 2      # n_y, column_index first
    assert d["layout"]["dim"][1]["size"] == 4      # n_x, row_index second
    assert len(d["data"]) == 8

    g2 = GridMap(2.0, 1.0, 0.5)
    g2.layers["elevation"][0, 0] = 7.0
    g2.layers["elevation"][1, 0] = 8.0
    flat = g2.to_msg()["data"][0]["data"]
    assert flat[0] == 7.0 and flat[1] == 8.0, "not column-major"

    assert abs(cost_to_traversability(1.0) - 0.5) < 1e-12
    assert cost_to_traversability(0.0) == 1.0
    assert cost_to_traversability(2.0) == 0.0
    print("gridmap selftest ok")


if __name__ == "__main__":
    selftest()
