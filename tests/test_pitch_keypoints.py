from football_analytics.pitch import DEFAULT_PITCH_LANDMARKS


def test_canonical_landmarks_are_unique_and_inside_pitch() -> None:
    ids = [landmark.id for landmark in DEFAULT_PITCH_LANDMARKS]
    coordinates = [(landmark.pitch_x, landmark.pitch_y) for landmark in DEFAULT_PITCH_LANDMARKS]

    assert len(DEFAULT_PITCH_LANDMARKS) == 32
    assert ids == [f"{index:02d}" for index in range(1, 33)]
    assert len(ids) == len(set(ids))
    assert len(coordinates) == len(set(coordinates))
    assert all(0 <= x <= 105 and 0 <= y <= 68 for x, y in coordinates)
