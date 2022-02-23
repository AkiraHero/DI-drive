from easydict import EasyDict

ppo_config = dict(
    exp_name='ppo21_bev32_lr1e4_bs128_ns3000_update5_train_ft',
    env=dict(
        collector_env_num=15,
        evaluator_env_num=1,
        simulator=dict(
            town='Town01',
            delta_seconds=0.05,
            disable_two_wheels=True,
            verbose=False,
            waypoint_num=32,
            planner=dict(
                type='behavior',
                resolution=1,
            ),
            obs=(
                dict(
                    name='birdview',
                    type='bev',
                    size=[160, 160],
                    pixels_per_meter=5,
                    pixels_ahead_vehicle=100,
                ),
                dict(
                    name='toplidar',
                    type='lidar',
                    channels=64,
                    range=32,
                    points_per_second=1280000,
                    rotation_frequency=20,
                    upper_fov=10.0,
                    lower_fov=-45.0,
                    position=[0, 0.0, 1.6],
                    rotation=[0, 0, 0],
                    fixed_pt_num=40000,
                ),
            ),
        ),
        enable_detector=False,
        detector=dict(
            model_repo="openpcdet",
            model_name="pointpillar",
            ckpt="/cpfs2/user/juxiaoliang/checkpoint_epoch_160.pth",
            # ckpt="/home/xlju/Downloads/pointpillar/pointpillar/ckpt/checkpoint_epoch_160.pth",
            max_batch_size=32,
            data_config=dict(
                class_names=['Car', 'Pedestrian'],
                point_feature_encoder=dict(
                    num_point_features=4,
                ),
                depth_downsample_factor=None
            ),
            score_thres={
                "vehicle": 0.6,
                "walker": 0.5
            }
        ),
        col_is_failure=True,
        stuck_is_failure=True,
        ignore_light=True,
        ran_light_is_failure=False,
        off_road_is_failure=True,
        wrong_direction_is_failure=True,
        off_route_is_failure=True,
        off_route_distance=7.5,
        reward_func="customized_compute_reward",
        #reward_type=['goal', 'distance', 'speed', 'angle', 'failure', 'lane'],
        success_distance=2.0,

        replay_path='./ppo_video',
        visualize=dict(
            type='birdview',
            # save_dir="see_collection",
            # outputs=['video'],
        ),
        manager=dict(
            collect=dict(
                auto_reset=True,
                shared_memory=False,
                context='spawn',
                max_retry=1,
                retry_type='renew',
                step_timeout=120,
                reset_timeout=120,
            ),
            eval=dict(
                auto_reset=True,
                shared_memory=False,
                context='spawn',
                max_retry=1,
                retry_type='renew',
                step_timeout=120,
                reset_timeout=120,
            )
        ),
        wrapper=dict(
            collect=dict(suite='train_akira_short_turn_nocar', ),
            eval=dict(suite='train_akira_short_turn_nocar', ),
        ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[9000, 9032, 2]),
    ],
    policy=dict(
        cuda=True,
        nstep_return=False,
        on_policy=True,
        model=dict(
            obs_shape=[6, 160, 160],
        ),
        learn=dict(
            epoch_per_collect=5,
            batch_size=128,
            learning_rate=0.0001,
            weight_decay=0.0001,
            value_weight=0.5,
            adv_norm=False,
            entropy_weight=0.01,
            clip_ratio=0.2,
            target_update_freq=100,
            learner=dict(
                hook=dict(
                    log_show_after_iter=1000,
                    #load_ckpt_before_run='/cpfs2/user/juxiaoliang/project/DI-drive/noisy_planning/output_log/ppo-shortturn_gamma0.995-2022-02-22-04-47-02/ckpt/ckpt_interrupt.pth.tar',
                    save_ckpt_after_iter=3000,
                ),
            ),
        ),
        collect=dict(
            pre_sample_num=3000,
            n_sample=3000,
            collector=dict(
                collect_print_freq=500,
                deepcopy_obs=True,
                transform_obs=True,
            ),
            discount_factor=0.995,
            gae_lambda=0.95,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=5000,
                n_episode=5,
                stop_rate=0.7,
                transform_obs=True,
            ),
        ),
    ),
)

default_train_config = EasyDict(ppo_config)
