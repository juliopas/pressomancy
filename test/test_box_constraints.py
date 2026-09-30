from .create_system import sim_inst, BoxTestCase
import numpy as np


class BoxWallsTest(BoxTestCase):
    """Generic walls (all six, or one side), a custom top wall with per-type WCA, and every removal mode."""

    box_dim = (16, 16, 16)

    def setUp(self):
        super().setUp()
        self.box_len = sim_inst.sys.box_l
        for typ, sigma in ((1, 1), (2, 2), (3, 3)):
            sim_inst.sys.part.add(pos=(0, 0, 0), type=typ)
            sim_inst.sys.non_bonded_inter[typ, typ].wca.set_params(epsilon=1, sigma=sigma)

    def test_generic_box_walls(self):
        """Each wall matches its expected (dist, normal); the returned list is `sys.constraints`; removal leaves none."""
        six = [
            (0, [0, 0, 1]),                     # bottom
            (-self.box_len[2], [0, 0, -1]),     # top
            (0, [0, 1, 0]),                     # left
            (-self.box_len[1], [0, -1, 0]),     # right
            (0, [1, 0, 0]),                     # back
            (-self.box_len[0], [-1, 0, 0]),     # front
        ]
        for sides, expected in (('all', six), ('top', six[1:2])):
            with self.subTest(sides=sides):
                box_constraints = sim_inst.add_box_constraints(sides=sides)
                self.assertEqual(list(box_constraints), list(sim_inst.sys.constraints))
                for wall, (exp_dist, exp_normal) in zip(box_constraints, expected, strict=True):
                    np.testing.assert_allclose(wall.shape.dist, exp_dist)
                    np.testing.assert_allclose(np.copy(wall.shape.normal), exp_normal)
                sim_inst.remove_box_constraints()
                self.assertEqual(list(sim_inst.sys.constraints), [])

    def test_custom_wall_with_wca(self):
        """A custom top wall plus per-type WCA with the wall type; removed by type, by wall, and all."""
        inter = sim_inst.sys.non_bonded_inter
        top_position = self.box_len[2] / 2
        box_constraints = sim_inst.add_box_constraints(
            wall_type=0, sides='no-sides', top=top_position, inter='wca', types_=(1, 2, 3))
        wall_bottom, wall_top = box_constraints
        np.testing.assert_allclose(wall_top.shape.dist, -top_position)
        np.testing.assert_allclose(np.copy(wall_top.shape.normal), [0, 0, -1])

        # (sigma, epsilon > 0) of each type's WCA with the wall type, after every step
        def wall_wca():
            return [(inter[t, 0].wca.sigma, inter[t, 0].wca.epsilon > 0) for t in (1, 2, 3)]
        on = [(inter[t, t].wca.sigma / 2, True) for t in (1, 2, 3)]
        off = (0, False)
        self.assertEqual(wall_wca(), on)

        sim_inst.remove_box_constraints(part_types=1)
        self.assertEqual(wall_wca(), [off, *on[1:]])

        sim_inst.remove_box_constraints(wall_top)
        self.assertEqual(list(sim_inst.sys.constraints), [wall_bottom])
        self.assertEqual(wall_wca(), [off, *on[1:]])

        sim_inst.remove_box_constraints()
        self.assertEqual(list(sim_inst.sys.constraints), [])
        self.assertEqual(wall_wca(), [off] * 3)
