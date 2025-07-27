import numpy as np

from pypso.util import terrain


def test_is_colliding_with_heightfield():
    # 创建一个简单的高度场 f(x, y) = -(x^2 + y^2) + 50
    x = np.linspace(-5, 5, 3)
    y = np.linspace(-5, 5, 3)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = -(x_grid ** 2 + y_grid ** 2) + 50
    print(z_grid)

    # 测试点在高度场上方
    point_above = np.array([0, 0, 55])
    assert not terrain.is_collision_detected(point_above, x_grid, y_grid, z_grid)

    # 测试点在高度场下方
    point_below = np.array([0, 0, 45])
    assert terrain.is_collision_detected(point_below, x_grid, y_grid, z_grid)

    # 测试点就在高度场上
    point_on = np.array([0, 0, 50])
    assert terrain.is_collision_detected(point_on, x_grid, y_grid, z_grid)

    # 测试点在高度场边界
    point_edge = np.array([-5, -5, 0])
    assert not terrain.is_collision_detected(point_edge, x_grid, y_grid, z_grid)
