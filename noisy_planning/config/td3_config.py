from easydict import EasyDict

td3_config = dict(
    exp_name='td32_bev32_buf4e5_lr1e4_bs128_ns3000_update4_train_ft',
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
            )
        ),
        enable_detector=True,
        detector=dict(
            model_repo="openpcdet",
            model_name="pointpillar",
            ckpt="/cpfs2/user/juxiaoliang/checkpoint_epoch_160.pth",
            #ckpt="/home/xlju/Downloads/pointpillar/pointpillar/ckpt/checkpoint_epoch_160.pth",
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
        # finish_reward=300,
        replay_path='./td3_video',
        visualize=dict(
            type='birdview',
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
            collect=dict(suite='train_akira', ),
            eval=dict(suite='eval_akira', ),
        ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[9000, 9032, 2]),
    ],
    policy=dict(
        cuda=True,
        priority=True,
        model=dict(
            obs_shape=[5, 160, 160],
        ),
        learn=dict(
            batch_size=64,
            learning_rate_actor=0.0001,
            learning_rate_critic=0.0001,
            weight_decay=0.0001,
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=1000,
                    save_ckpt_after_iter=3000,
                ),
            ),
        ),
        collect=dict(
            pre_sample_num=3000,
            noise_sigma=0.1,
            n_sample=3000,
            collector=dict(
                collect_print_freq=1000,
                deepcopy_obs=True,
                transform_obs=True,
            ),
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=3000,
                n_episode=5,
                stop_rate=2.0, # do not stop by eval
                transform_obs=True,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=400000,
                max_use=16,
                monitor=dict(
                    sampled_data_attr=dict(
                        print_freq=100,  # times
                    ),
                    periodic_thruput=dict(
                        seconds=120,
                    ),
                ),
            ),
        ),
    ),
)

default_train_config = EasyDict(td3_config)
