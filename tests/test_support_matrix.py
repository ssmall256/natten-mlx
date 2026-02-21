import natten_mlx


def test_support_matrix_shape():
    matrix = natten_mlx.get_support_matrix()
    assert set(matrix.keys()) == {"pure", "fast_metal", "nanobind"}
    for backend_name, row in matrix.items():
        assert isinstance(row["available"], bool), backend_name
        assert set(row["forward"].keys()) == {"na1d", "na2d", "na3d", "split_qk_av"}
        assert set(row["backward"].keys()) == {"na1d", "na2d", "na3d", "split_qk_av"}
        assert set(row["fusion"].keys()) == {"na1d", "na2d", "na3d"}
        assert isinstance(row["constraints"], list)

    assert matrix["fast_metal"]["fusion"] == {"na1d": True, "na2d": True, "na3d": True}
    assert matrix["nanobind"]["fusion"] == {"na1d": True, "na2d": True, "na3d": True}
