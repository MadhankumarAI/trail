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
    traversability_slope        per-criterion sublayers, named after the
    traversability_step         criteria grid_map's own demo chain combines
    traversability_roughness    (slope and roughness), plus step
    obstacle_height             metres a static obstacle stands above ground

The traversability convention is inverted from our cost. grid_map's own demo
filter chain (grid_map_demos/config/filters_demo_filter_chain.yaml) defines it
as

    0.5 * (1 - slope/0.6) + 0.5 * (1 - roughness/0.1)

clamped to [0, 1] -- that is, each criterion enters as `1 - x/limit`, reaching
0 exactly at its limit. Ours uses the same per-criterion form:

    traversability = clip(1 - score, 0, 1)

with one deliberate difference. They combine criteria by weighted MEAN; our
score is the worst normalised violation, a MAX. The mean lets a gentle slope
average away lethal roughness, which for a ground vehicle is the wrong
direction to be wrong in. Both agree at the limit: score 1.0 -> 0.0.

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

    Matches the `1 - x/limit` form of grid_map's demo filter chain, so a cell
    at the vehicle limit reports 0 exactly as theirs does.
    """
    return np.clip(1.0 - np.asarray(score, float), 0.0, 1.0)


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
        """(ix, iy) for world positions, and a mask of what is inside.

        Transcribed from GridMapMath.cpp getIndexFromPosition, which does NOT
        use the same offset as getPositionFromIndex -- a trap worth spelling
        out. Position-from-index uses getVectorToFirstCell (0.5*L - 0.5*res,
        the CENTRE of the first cell); index-from-position uses
        getVectorToOrigin (0.5*L, the map EDGE) and then truncates on the cast
        to int. Using the cell-centre offset with rounding agrees almost
        everywhere and disagrees exactly on cell boundaries, which is the kind
        of off-by-one that only shows up as a seam.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        # indexVector = (position - 0.5*L - mapPosition) / res, then negated by
        # transformMapFrameToBufferOrder and truncated by the cast to Index
        vx = (x - 0.5 * self.length_x - self.position[0]) / self.resolution
        vy = (y - 0.5 * self.length_y - self.position[1]) / self.resolution
        ix = np.trunc(-vx).astype(int)
        iy = np.trunc(-vy).astype(int)
        # checkIfPositionWithinMap: the transformed position must be in
        # [0, length), which makes the map (centre - L/2, centre + L/2]
        ok = ((-vx * self.resolution >= 0.0)
              & (-vx * self.resolution < self.length_x)
              & (-vy * self.resolution >= 0.0)
              & (-vy * self.resolution < self.length_y)
              & (ix >= 0) & (ix < self.n_x) & (iy >= 0) & (iy < self.n_y))
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
        # getIndexShiftFromPositionShift rounds half AWAY from zero, not to
        # even -- np.rint would disagree on exact half cells
        v = (new_position - self.position) / self.resolution
        shift = np.trunc(v + 0.5 * np.where(v > 0, 1.0, -1.0)).astype(int)
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
            # GridMapRosConverter.cpp: outer_start_index = getStartIndex()(0),
            # inner_start_index = getStartIndex()(1). The msg comments call
            # them "row" and "column" start index respectively, which reads
            # backwards against the data layout's dim labels; the code is the
            # authority. move() here rolls the data instead of rotating a
            # buffer origin, so both stay zero.
            "outer_start_index": int(self.start[0]),
            "inner_start_index": int(self.start[1]),
        }

    # ------------------------------------------------------------- memory
    def memory_bytes(self):
        """What grid_map actually holds: every cell of every layer, always."""
        return self.n_x * self.n_y * len(self.layer_names) * 4


def _ref_index_from_position(px, py, length_x, length_y, pos_x, pos_y, res):
    """GridMapMath.cpp getIndexFromPosition, transcribed one line at a time.

    Deliberately scalar, ugly and literal. It exists only to disagree with the
    vectorised version if that ever drifts, so it must not share any of its
    reasoning -- the C++ is reproduced statement by statement:

        getVectorToOrigin(offset, mapLength)        -> offset = 0.5 * length
        indexVector = (position - offset - mapPosition) / resolution
        index = transformMapFrameToBufferOrder(indexVector)   -> negate
                then cast to int, which truncates
    """
    off_x, off_y = 0.5 * length_x, 0.5 * length_y
    ivx = (px - off_x - pos_x) / res
    ivy = (py - off_y - pos_y) / res
    return int(-ivx), int(-ivy)          # int() truncates, as the C++ cast does


def _ref_position_from_index(ix, iy, length_x, length_y, pos_x, pos_y, res):
    """GridMapMath.cpp getPositionFromIndex, likewise literal.

        getVectorToFirstCell(offset, ...)  -> 0.5 * length - 0.5 * resolution
        position = mapPosition + offset + resolution * (-index)
    """
    off_x = 0.5 * length_x - 0.5 * res
    off_y = 0.5 * length_y - 0.5 * res
    return pos_x + off_x + res * (-ix), pos_y + off_y + res * (-iy)


def selftest():
    """Check the geometry against a literal transcription of the C++.

    A mirrored or half-cell-shifted map is the failure mode here, and it looks
    entirely fine on screen.
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

    # agree with the transcribed C++ everywhere, including exactly on cell
    # boundaries, which is where the old cell-centre-plus-rounding version
    # silently differed by one
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(-8.0, 14.0, 4000),
                        np.arange(-7.0, 13.0, 0.5),      # exact boundaries
                        np.arange(-7.0, 13.0, 0.25)])    # exact centres
    y = np.concatenate([rng.uniform(-7.0, 5.0, 4000),
                        np.arange(-6.0, 4.0, 0.5)[:40],
                        np.arange(-6.0, 4.0, 0.25)[:80]])
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    ix, iy, ok = g.index_from_position(x, y)
    bad = 0
    for j in range(n):
        rx, ry = _ref_index_from_position(x[j], y[j], g.length_x, g.length_y,
                                          g.position[0], g.position[1],
                                          g.resolution)
        if (rx, ry) != (int(ix[j]), int(iy[j])):
            bad += 1
    assert bad == 0, f"{bad}/{n} indices disagree with the transcribed C++"

    for j in range(0, n, 37):
        if not ok[j]:
            continue
        rpx, rpy = _ref_position_from_index(ix[j], iy[j], g.length_x,
                                            g.length_y, g.position[0],
                                            g.position[1], g.resolution)
        bx, by = g.position_from_index(ix[j], iy[j])
        assert abs(rpx - bx) < 1e-9 and abs(rpy - by) < 1e-9

    # and the recovered centre must be within half a cell of the query
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

    # start indices follow GridMapRosConverter, outer = axis 0
    m2 = GridMap(2.0, 1.0, 0.5)
    m2.start = np.array([3, 5])
    msg2 = m2.to_msg()
    assert msg2["outer_start_index"] == 3 and msg2["inner_start_index"] == 5

    # grid_map's own filter chain form: 1 - x/limit, zero AT the limit
    assert cost_to_traversability(0.0) == 1.0
    assert cost_to_traversability(1.0) == 0.0
    assert cost_to_traversability(2.0) == 0.0
    assert abs(cost_to_traversability(0.25) - 0.75) < 1e-12
    print("gridmap selftest ok")


if __name__ == "__main__":
    selftest()
