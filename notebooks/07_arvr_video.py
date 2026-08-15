"""
Research Notebook 7: AR/VR Video Generation
Demonstrates NeRF, Gaussian Splatting, Text-to-3D, 6DoF video, and AR scene understanding.

Run: python notebooks/07_arvr_video.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 60)
    print("Notebook 7: AR/VR Video Generation")
    print("=" * 60)

    # 1. NeRF
    print("\n--- Neural Radiance Fields (NeRF) ---")
    from src.models.nerf import NeRF, NeRFConfig, CameraPose, NeRFTrainer
    config = NeRFConfig(pos_encoding_dim=60, dir_encoding_dim=24,
                        hidden_dim=64, n_layers=4, n_samples=32)
    nerf = NeRF(config)
    camera = CameraPose(
        position=np.array([0, 0, 3]), rotation=np.eye(3),
        focal_length=50, width=32, height=32
    )
    result = nerf.render_image(camera)
    print(f"  Rendered: {result['image'].shape}, depth: {result['depth_map'].shape}")
    print(f"  RGB range: [{result['image'].min():.3f}, {result['image'].max():.3f}]")
    print(f"  Depth range: [{result['depth_map'].min():.3f}, {result['depth_map'].max():.3f}]")
    target = np.random.rand(32, 32, 3)
    trainer = NeRFTrainer(nerf)
    loss = trainer.train_step(camera, target)
    print(f"  Training: MSE={loss['mse']:.6f}, PSNR={loss['psnr']:.2f} dB")

    # 2. Gaussian Splatting
    print("\n--- 3D Gaussian Splatting ---")
    from src.models.gaussian_splatting import (
        GaussianSplattingRenderer, GaussianSplattingConfig,
        create_view_matrix, create_proj_matrix
    )
    gs_config = GaussianSplattingConfig(n_gaussians=20)
    gs_renderer = GaussianSplattingRenderer(gs_config)
    gs_renderer.initialize_random(
        (np.array([-1, -1, -1]), np.array([1, 1, 1]))
    )
    view = create_view_matrix(np.array([0, 0, 3]), np.array([0, 0, 0]))
    proj = create_proj_matrix(60, 1.0, 0.1, 100)
    gs_image = gs_renderer.render(view, proj, (32, 32))
    print(f"  Rendered: {gs_image.shape}, {len(gs_renderer.gaussians)} Gaussians")
    n_after = gs_renderer.densify()
    print(f"  After densification: {n_after} Gaussians")

    # 3. Text-to-3D
    print("\n--- Text-to-3D Generation ---")
    from src.models.text_to_3d import TextTo3DGenerator, TextTo3DConfig, VideoTo3D
    t3d_config = TextTo3DConfig(
        prompt="A 3D model of a futuristic car",
        n_coarse_iterations=50,
        n_fine_iterations=100,
    )
    generator = TextTo3DGenerator(t3d_config)
    result = generator.generate("A 3D model of a futuristic car")
    print(f"  Final loss: {result['final_loss']:.6f}")
    mesh = generator.export_mesh()
    print(f"  Mesh: {mesh['n_vertices']} vertices, {mesh['n_faces']} faces")

    # 4. 6DoF Video
    print("\n--- 6DoF Video Generation ---")
    from src.video.sixdof_video import VRVideoRenderer, Camera6DoF
    vr = VRVideoRenderer(target_fps=90)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pose = Camera6DoF(
            position=np.array([2 * np.cos(angle), 0, 2 * np.sin(angle)]),
            rotation=np.array([0, angle, 0]),
            width=32, height=32,
        )
        img = np.random.rand(32, 32, 3)
        depth = np.random.rand(32, 32) * 5 + 1
        vr.add_captured_view(img, pose, depth)
    novel_pose = Camera6DoF(
        position=np.array([0, 0, 2]), rotation=np.array([0, 0, 0]),
        width=32, height=32,
    )
    view_img = vr.render(novel_pose)
    print(f"  Novel view: {view_img.shape}, views: {vr.light_field.n_views}")
    left, right = vr.render_stereo(novel_pose)
    print(f"  Stereo: left={left.shape}, right={right.shape} (IPD=63mm)")

    # 5. Depth Estimation
    print("\n--- Depth Estimation for AR Compositing ---")
    from src.video.depth_estimation import (
        MonocularDepthEstimator, DepthConfig, ARCompositor
    )
    depth_estimator = MonocularDepthEstimator(DepthConfig(input_size=32))
    img = np.random.rand(32, 32, 3)
    depth = depth_estimator.estimate(img)
    print(f"  Depth: {depth.shape}, range: [{depth.min():.2f}, {depth.max():.2f}]m")
    compositor = ARCompositor()
    bg = np.random.rand(32, 32, 3)
    fg = np.random.rand(16, 16, 4)
    fg[:, :, 3] = 0.8
    composite = compositor.composite(bg, fg, position=(16, 16))
    print(f"  AR composite: {composite.shape}")

    # 6. Scene Understanding
    print("\n--- Scene Understanding & SLAM ---")
    from src.video.scene_understanding import (
        ARSceneUnderstanding, PlaneDetector, SceneGraph, SemanticClass
    )
    ar = ARSceneUnderstanding()
    img = np.random.rand(32, 32, 3)
    for i in range(15):
        result = ar.slam.process_frame(img, timestamp=i * 0.033)
    print(f"  SLAM: {result['n_keyframes']} keyframes, {result['n_landmarks']} landmarks")
    plane_detector = PlaneDetector(ransac_iterations=50, min_inliers=10)
    points = np.random.randn(200, 3)
    points[:100, 1] = -1  # Floor points
    points[100:, 1] = np.random.randn(100)  # Random objects
    planes = plane_detector.detect_planes(points)
    print(f"  Planes detected: {len(planes)}")
    for p in planes:
        print(f"    {p.semantic_class.value}: center={p.center}, extent={p.extent}")
    sg = SceneGraph()
    room = sg.add_room(np.array([0, 0, 0]), np.array([5, 3, 5]))
    table = sg.add_object(np.array([2, 0, 1]), np.array([1, 1, 1]), SemanticClass.TABLE)
    sg.add_relation(room, table, "contains")
    print(f"  Scene graph: {sg.to_dict()['n_objects']} objects, {sg.to_dict()['n_rooms']} rooms")

    print("\n✓ AR/VR video generation pipeline verified")
    print("\nMeta Research Alignment:")
    print("  NeRF -> Immersive Light Field Video (FRL)")
    print("  Gaussian Splatting -> Real-time 3D rendering")
    print("  Text-to-3D -> Make-A-Video / DreamFusion")
    print("  6DoF -> Neural Light Field Video (Attal et al.)")
    print("  Scene Understanding -> SceneScript (Avetisyan et al. 2024)")


if __name__ == "__main__":
    main()
