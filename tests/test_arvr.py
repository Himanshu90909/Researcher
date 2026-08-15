"""
Tests for AR/VR video generation modules.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_nerf():
    """Test NeRF rendering."""
    from src.models.nerf import NeRF, NeRFConfig, CameraPose
    config = NeRFConfig(pos_encoding_dim=60, dir_encoding_dim=24,
                        hidden_dim=64, n_layers=4, n_samples=16)
    nerf = NeRF(config)
    camera = CameraPose(
        position=np.array([0, 0, 3]), rotation=np.eye(3),
        focal_length=50, width=16, height=16
    )
    result = nerf.render_image(camera)
    assert result['image'].shape == (16, 16, 3)
    assert result['depth_map'].shape == (16, 16)
    assert np.all(result['image'] >= 0) and np.all(result['image'] <= 1)
    print("✓ NeRF: volume rendering, depth estimation")


def test_gaussian_splatting():
    """Test Gaussian Splatting renderer."""
    from src.models.gaussian_splatting import (
        GaussianSplattingRenderer, GaussianSplattingConfig,
        create_view_matrix, create_proj_matrix, Gaussian3D
    )
    config = GaussianSplattingConfig(n_gaussians=10)
    renderer = GaussianSplattingRenderer(config)
    renderer.gaussians.append(Gaussian3D(
        position=np.array([0, 0, 0]),
        scale=np.array([0.05, 0.05, 0.05]),
        rotation=np.array([1, 0, 0, 0]),
        color=np.array([0.5, 0.3, 0.2]),
        opacity=0.8,
    ))
    view = create_view_matrix(np.array([0, 0, 3]), np.array([0, 0, 0]))
    proj = create_proj_matrix(60, 1.0, 0.1, 100)
    image = renderer.render(view, proj, (16, 16))
    assert image.shape == (16, 16, 3)
    assert np.all(image >= 0) and np.all(image <= 1)
    stats = renderer.get_stats()
    assert stats['n_gaussians'] == 1
    print("✓ Gaussian Splatting: splatting, rendering, stats")


def test_text_to_3d():
    """Test text-to-3D generation."""
    from src.models.text_to_3d import TextTo3DGenerator, TextTo3DConfig, VideoTo3D
    config = TextTo3DConfig(n_coarse_iterations=10, n_fine_iterations=20)
    gen = TextTo3DGenerator(config)
    result = gen.generate("A 3D chair")
    assert result['n_iterations'] == 30
    assert result['final_loss'] >= 0
    mesh = gen.export_mesh()
    assert mesh['n_vertices'] > 0
    print("✓ Text-to-3D: SDS optimization, mesh export")


def test_video_to_3d():
    """Test video-to-3D reconstruction."""
    from src.models.text_to_3d import VideoTo3D
    v2_3d = VideoTo3D(n_keyframes=5)
    frames = [np.random.rand(32, 32, 3) for _ in range(15)]
    result = v2_3d.reconstruct(frames)
    assert result['n_keyframes'] == 5
    assert result['n_points'] > 0
    assert len(result['poses']) == 5
    print("✓ Video-to-3D: keyframe extraction, pose estimation, reconstruction")


def test_sixdof_video():
    """Test 6DoF video rendering."""
    from src.video.sixdof_video import VRVideoRenderer, Camera6DoF
    vr = VRVideoRenderer(target_fps=90)
    for i in range(4):
        angle = 2 * np.pi * i / 4
        pose = Camera6DoF(
            position=np.array([2 * np.cos(angle), 0, 2 * np.sin(angle)]),
            rotation=np.array([0, angle, 0]),
            width=16, height=16,
        )
        img = np.random.rand(16, 16, 3)
        depth = np.random.rand(16, 16) * 5 + 1
        vr.add_captured_view(img, pose, depth)
    novel = Camera6DoF(
        position=np.array([0, 0, 2]), rotation=np.array([0, 0, 0]),
        width=16, height=16,
    )
    view = vr.render(novel)
    assert view.shape == (16, 16, 3)
    left, right = vr.render_stereo(novel)
    assert left.shape == right.shape
    print("✓ 6DoF video: novel view synthesis, stereo rendering")


def test_depth_estimation():
    """Test depth estimation."""
    from src.video.depth_estimation import MonocularDepthEstimator, DepthConfig
    estimator = MonocularDepthEstimator(DepthConfig(input_size=16, n_bins=32))
    image = np.random.rand(16, 16, 3)
    depth = estimator.estimate(image)
    assert depth.shape == (16, 16)
    assert depth.min() >= 0
    adabins = estimator.estimate_adabins(image)
    assert adabins['n_bins'] == 32
    print("✓ Depth estimation: monocular, AdaBins")


def test_ar_compositing():
    """Test AR compositing."""
    from src.video.depth_estimation import ARCompositor
    compositor = ARCompositor()
    bg = np.random.rand(16, 16, 3)
    fg = np.random.rand(8, 8, 4)
    fg[:, :, 3] = 0.5
    result = compositor.composite(bg, fg, position=(8, 8))
    assert result.shape == (16, 16, 3)
    print("✓ AR compositing: depth-aware occlusion")


def test_plane_detection():
    """Test 3D plane detection."""
    from src.video.scene_understanding import PlaneDetector, SemanticClass
    detector = PlaneDetector(ransac_iterations=50, min_inliers=10)
    # Floor plane at y=-1
    floor = np.array([[x, -1, z] for x in range(-5, 5) for z in range(-5, 5)])
    # Random objects
    objects = np.random.randn(50, 3)
    points = np.vstack([floor.astype(float), objects])
    planes = detector.detect_planes(points)
    assert len(planes) > 0
    floor_planes = [p for p in planes if p.semantic_class == SemanticClass.FLOOR]
    assert len(floor_planes) > 0
    print(f"✓ Plane detection: {len(planes)} planes, {len(floor_planes)} floor")


def test_scene_graph():
    """Test scene graph construction."""
    from src.video.scene_understanding import SceneGraph, SemanticClass
    sg = SceneGraph()
    room = sg.add_room(np.array([0, 0, 0]), np.array([5, 3, 5]))
    table = sg.add_object(np.array([2, 0, 1]), np.array([1, 1, 1]), SemanticClass.TABLE)
    chair = sg.add_object(np.array([2, 0, 2]), np.array([0.5, 1, 0.5]), SemanticClass.CHAIR)
    sg.add_relation(room, table, "contains")
    sg.add_relation(table, chair, "near")
    graph = sg.to_dict()
    assert graph['n_objects'] == 2
    assert graph['n_rooms'] == 1
    assert graph['n_relations'] == 2
    tables = sg.get_objects_by_class(SemanticClass.TABLE)
    assert len(tables) == 1
    print("✓ Scene graph: objects, rooms, relations")


def test_slam():
    """Test SLAM pipeline."""
    from src.video.scene_understanding import SLAM
    slam = SLAM(n_features=50)
    img = np.random.rand(16, 16, 3)
    for i in range(20):
        result = slam.process_frame(img, timestamp=i * 0.033)
    assert result['n_keyframes'] >= 1
    assert result['n_landmarks'] > 0
    map_data = slam.get_map()
    assert map_data['n_points'] > 0
    print(f"✓ SLAM: {map_data['n_keyframes']} keyframes, {map_data['n_points']} landmarks")


if __name__ == "__main__":
    print("Running AR/VR Video Tests\n" + "=" * 40)
    test_nerf()
    test_gaussian_splatting()
    test_text_to_3d()
    test_video_to_3d()
    test_sixdof_video()
    test_depth_estimation()
    test_ar_compositing()
    test_plane_detection()
    test_scene_graph()
    test_slam()
    print("\n✅ All AR/VR tests passed!")
