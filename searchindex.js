Search.setIndex({docnames:["api_doc/data","api_doc/envs","api_doc/eval","api_doc/index","api_doc/models","api_doc/policy","api_doc/simulators","api_doc/utils","faq/index","features/carla_benchmark","features/casezoo","features/datasets","features/index","features/policy_feature","features/simulator_feature","features/visualize","index","installation/index","model_zoo/cict","model_zoo/cilrs","model_zoo/implicit","model_zoo/index","model_zoo/latent_rl","model_zoo/lbc","model_zoo/simple_rl","tutorial/auto_run","tutorial/carla_tutorial","tutorial/core_concepts","tutorial/il_tutorial","tutorial/index","tutorial/rl_tutorial"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api_doc/data.rst","api_doc/envs.rst","api_doc/eval.rst","api_doc/index.rst","api_doc/models.rst","api_doc/policy.rst","api_doc/simulators.rst","api_doc/utils.rst","faq/index.rst","features/carla_benchmark.rst","features/casezoo.rst","features/datasets.rst","features/index.rst","features/policy_feature.rst","features/simulator_feature.rst","features/visualize.rst","index.rst","installation/index.rst","model_zoo/cict.rst","model_zoo/cilrs.rst","model_zoo/implicit.rst","model_zoo/index.rst","model_zoo/latent_rl.rst","model_zoo/lbc.rst","model_zoo/simple_rl.rst","tutorial/auto_run.rst","tutorial/carla_tutorial.rst","tutorial/core_concepts.rst","tutorial/il_tutorial.rst","tutorial/index.rst","tutorial/rl_tutorial.rst"],objects:{"core.data.benchmark_dataset_saver":{BenchmarkDatasetSaver:[0,0,1,""]},"core.data.benchmark_dataset_saver.BenchmarkDatasetSaver":{make_dataset_path:[0,1,1,""],make_index:[0,1,1,""],save_episodes_data:[0,1,1,""]},"core.data.carla_benchmark_collector":{CarlaBenchmarkCollector:[0,0,1,""]},"core.data.carla_benchmark_collector.CarlaBenchmarkCollector":{close:[0,1,1,""],collect:[0,1,1,""],reset:[0,1,1,""]},"core.envs":{BaseCarlaEnv:[1,0,1,""],BenchmarkEnvWrapper:[1,0,1,""],CarlaEnvWrapper:[1,0,1,""],ScenarioCarlaEnv:[1,0,1,""],SimpleCarlaEnv:[1,0,1,""]},"core.envs.BaseCarlaEnv":{close:[1,1,1,""],reset:[1,1,1,""],seed:[1,1,1,""],step:[1,1,1,""]},"core.envs.BenchmarkEnvWrapper":{reset:[1,1,1,""],step:[1,1,1,""]},"core.envs.CarlaEnvWrapper":{info:[1,1,1,""],reset:[1,1,1,""],step:[1,1,1,""]},"core.envs.ScenarioCarlaEnv":{close:[1,1,1,""],compute_reward:[1,1,1,""],get_observations:[1,1,1,""],is_failure:[1,1,1,""],is_success:[1,1,1,""],render:[1,1,1,""],reset:[1,1,1,""],seed:[1,1,1,""],step:[1,1,1,""]},"core.envs.SimpleCarlaEnv":{close:[1,1,1,""],compute_reward:[1,1,1,""],get_observations:[1,1,1,""],is_failure:[1,1,1,""],is_success:[1,1,1,""],render:[1,1,1,""],reset:[1,1,1,""],seed:[1,1,1,""],step:[1,1,1,""]},"core.eval":{CarlaBenchmarkEvaluator:[2,0,1,""],SerialEvaluator:[2,0,1,""],SingleCarlaEvaluator:[2,0,1,""]},"core.eval.CarlaBenchmarkEvaluator":{close:[2,1,1,""],eval:[2,1,1,""],reset:[2,1,1,""],should_eval:[2,1,1,""]},"core.eval.SerialEvaluator":{close:[2,1,1,""],eval:[2,1,1,""],reset:[2,1,1,""],should_eval:[2,1,1,""]},"core.eval.SingleCarlaEvaluator":{close:[2,1,1,""],eval:[2,1,1,""]},"core.models":{BEVSpeedConvEncoder:[4,0,1,""],MPCController:[4,0,1,""],VehiclePIDController:[4,0,1,""]},"core.models.BEVSpeedConvEncoder":{forward:[4,1,1,""]},"core.models.MPCController":{forward:[4,1,1,""]},"core.models.VehiclePIDController":{forward:[4,1,1,""]},"core.models.model_wrappers":{SteerNoiseWrapper:[4,0,1,""]},"core.models.model_wrappers.SteerNoiseWrapper":{forward:[4,1,1,""]},"core.models.vae_model":{VanillaVAE:[4,0,1,""]},"core.models.vae_model.VanillaVAE":{decode:[4,1,1,""],encode:[4,1,1,""],forward:[4,1,1,""],generate:[4,1,1,""],loss_function:[4,1,1,""],reparameterize:[4,1,1,""],sample:[4,1,1,""]},"core.policy":{AutoMPCPolicy:[5,0,1,""],AutoPIDPolicy:[5,0,1,""],CILRSPolicy:[5,0,1,""],LBCBirdviewPolicy:[5,0,1,""],LBCImagePolicy:[5,0,1,""]},"core.policy.AutoMPCPolicy":{_forward_collect:[5,1,1,""],_forward_eval:[5,1,1,""],_reset_collect:[5,1,1,""],_reset_eval:[5,1,1,""]},"core.policy.AutoPIDPolicy":{_forward_collect:[5,1,1,""],_forward_eval:[5,1,1,""],_reset_collect:[5,1,1,""],_reset_eval:[5,1,1,""]},"core.policy.LBCBirdviewPolicy":{_forward_eval:[5,1,1,""],_reset_eval:[5,1,1,""]},"core.policy.LBCImagePolicy":{_forward_eval:[5,1,1,""],_reset_eval:[5,1,1,""]},"core.policy.base_carla_policy":{BaseCarlaPolicy:[5,0,1,""]},"core.simulators":{CarlaScenarioSimulator:[6,0,1,""],CarlaSimulator:[6,0,1,""]},"core.simulators.CarlaScenarioSimulator":{clean_up:[6,1,1,""],end_scenario:[6,1,1,""],get_criteria:[6,1,1,""],init:[6,1,1,""],run_step:[6,1,1,""]},"core.simulators.CarlaSimulator":{apply_control:[6,1,1,""],apply_planner:[6,1,1,""],clean_up:[6,1,1,""],get_information:[6,1,1,""],get_navigation:[6,1,1,""],get_sensor_data:[6,1,1,""],get_state:[6,1,1,""],init:[6,1,1,""],run_step:[6,1,1,""]},"core.simulators.base_simulator":{BaseSimulator:[6,0,1,""]},"core.simulators.base_simulator.BaseSimulator":{apply_control:[6,1,1,""],run_step:[6,1,1,""]},"core.utils.env_utils.stuck_detector":{StuckDetector:[7,0,1,""]},"core.utils.env_utils.stuck_detector.StuckDetector":{clear:[7,1,1,""],tick:[7,1,1,""]},"core.utils.others.visualizer":{Visualizer:[7,0,1,""]},"core.utils.others.visualizer.Visualizer":{done:[7,1,1,""],init:[7,1,1,""],paint:[7,1,1,""],run_visualize:[7,1,1,""]},"core.utils.planner.basic_planner":{BasicPlanner:[7,0,1,""]},"core.utils.planner.basic_planner.BasicPlanner":{clean_up:[7,1,1,""],get_incoming_waypoint_and_direction:[7,1,1,""],get_waypoints_list:[7,1,1,""],run_step:[7,1,1,""],set_destination:[7,1,1,""],set_route:[7,1,1,""]},"core.utils.planner.behavior_planner":{BehaviorPlanner:[7,0,1,""]},"core.utils.planner.behavior_planner.BehaviorPlanner":{run_step:[7,1,1,""]},"core.utils.simulator_utils.sensor_utils":{CollisionSensor:[7,0,1,""],SensorHelper:[7,0,1,""],TrafficLightHelper:[7,0,1,""]},"core.utils.simulator_utils.sensor_utils.CollisionSensor":{clear:[7,1,1,""]},"core.utils.simulator_utils.sensor_utils.SensorHelper":{clean_up:[7,1,1,""],get_sensors_data:[7,1,1,""],setup_sensors:[7,1,1,""]},"core.utils.simulator_utils.sensor_utils.TrafficLightHelper":{tick:[7,1,1,""]}},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"]},objtypes:{"0":"py:class","1":"py:method"},terms:{"0":[0,1,4,6,7,9,14,15,17,18,19,20,24,25,26],"00024":18,"0003":18,"04":[17,30],"05":[18,19],"099":17,"1":[2,4,6,7,9,11,14,17,18,19,24,26],"10":[4,6,14,16,17,18,19,28],"100":[4,9,11,14,19,28],"1000":[14,18],"100000":18,"100carla":20,"11":[0,14],"1109":18,"12":[14,28],"126":8,"128":[18,19],"13":14,"14":14,"15":[14,19,25],"16":[17,30],"160":14,"168":26,"18":[9,17],"192":26,"1e":19,"1s":26,"2":[1,4,5,9,14,15,18,19,24,25,26,28,30],"20":[9,14],"200":[7,18,19],"2000":[14,20,26],"2002":26,"2018":19,"2018end":19,"2019":[9,19,23],"2021":[16,18],"2080ti":28,"20ghz":30,"21":9,"22":9,"23":9,"24":30,"2477":18,"2484":18,"25":[9,18,19,24],"256":[18,24],"28":16,"2d":[11,14],"3":[4,5,7,14,15,17,18,19,27,30],"30":18,"300":19,"3000":[18,24],"3061336":18,"32":[18,19],"320":14,"32g":[20,30],"32x32x5":24,"360":18,"37":18,"384":14,"3d":[11,14],"3e":18,"4":[4,14,17,18,19,20,24,26],"40":18,"400":[14,18,19],"42":9,"43":9,"44275":18,"45":[9,19],"46":9,"48carla":20,"4d":14,"4gb":17,"5":[4,11,14,18,19,24,25,26,30],"50":[8,9,18],"500":25,"5000":26,"5010":26,"512":18,"52583":18,"5e":18,"6":[14,17,18,24],"60":[1,18],"600":[19,25],"60carla":20,"64":18,"640":18,"65000":18,"7":[14,17,18,28],"70":9,"72":9,"75":[4,24],"8":[4,14,17,24,25,26,28,30],"80":[9,18],"800":[19,25],"86":9,"8700":30,"88":9,"8d":8,"9":[6,14,17,26],"90":[9,14,19],"9000":[1,6,18,19,25,26,28,30],"9002":[25,26,30],"9004":26,"9006":26,"9008":26,"9010":[18,19,26,28,30],"9014":26,"9016":[26,30],"9050":6,"92":9,"9329":19,"9338":19,"9361054":18,"99":9,"9900k":28,"999":18,"abstract":[1,6],"break":24,"case":[10,22,26],"class":[0,1,2,4,5,6,7,14,24],"default":[0,1,2,4,5,6,7,8,10,14,17,18,25,26,28,30],"export":19,"final":[1,4,5,7,14,23],"float":[1,2,4,6,7,11,14],"function":[0,2,4,13,18,27],"import":[10,17],"int":[0,1,2,4,5,6,7,11,14],"new":[1,5,8,10,27],"null":8,"return":[0,1,2,4,5,6,7,11,13,14,24],"short":[1,7,10],"static":6,"switch":[14,25,30],"true":[7,14,15,18,19,24,28,30],"try":[16,17],"var":4,"void":14,"while":[1,6,24,26],A:[1,6,11,14,15],And:1,As:[13,14],By:[2,5,18,26,30],For:[0,2,5,9,11,13,14,15,24,26,28,30],If:[0,1,6,7,8,14,17,20,24,25,28],In:[1,5,6,7,8,10,22,25,26,28,30],It:[0,1,2,4,5,6,7,9,10,11,14,16,17,18,19,20,24,25,26,27,28,30],Its:1,NOT:[1,6,26],On:24,One:[13,27],Or:[8,17],The:[0,1,2,4,5,6,7,8,9,10,11,14,15,16,18,19,20,23,24,25,26,27,28,30],Their:[5,11],Then:[4,5,14,15,17,23,25,26,28],There:[10,14],To:[1,5,14,17,19,25,28],WITH:2,Will:[2,14],_00000:11,_acc_list:24,_forward_collect:[5,13],_forward_ev:5,_interfac:5,_interface_xxx:5,_log:18,_preload:[18,19,28],_replac:24,_reset_collect:5,_reset_ev:5,_steer_list:24,a1:8,a2:8,a3:8,a4:8,a5:8,abil:10,abl:[6,10,13,20],about:[1,6,14,20,30],abov:[4,15,24,28],academ:16,academia:16,acc:[6,24],acceler:[11,14,18],acceleration_loss_weight:18,accord:[0,1,5,6,7,9,10,30],account:[7,14],accur:10,achiev:[6,19,20],across:16,act:13,action:[0,1,7,13,15,24,27],actor:[1,6,7,14,26],actual:6,ad:[1,8,14,15,16,22],adapt:16,add:[1,4,5,6,7,8,10,11,14,15,24,30],addit:[0,2],adrien:19,afford:[9,21],aforement:30,after:[0,13,30],again:[0,8],agent:[7,14,15,17,20,27],agent_st:14,aggress:7,ahead:[7,14],ahenb:23,aim:[1,10],al:9,alexei:19,algorithm:30,alias:9,align:1,all:[0,1,2,5,6,7,9,10,11,13,14,16,19,24,25,26,27,28,30],allow:[9,10,11,13,26],along:1,alreadi:6,also:[1,6,7,9,10,14,17,18,26,27,28,30],alwai:1,amazonaw:17,among:14,amount:[9,10,26],an:[0,1,4,5,6,7,9,10,11,13,14,15,16,19,20,23,25,26,27,28],anaconda:8,analyz:[7,27],angl:[4,6],ani:[0,1,2,4,5,6,7,10,16,17,24,27],antonio:19,anyth:8,aonfigur:9,api:[8,9,14,16,24,26],aplli:6,appli:[4,6,7,16,28],applic:16,apply_control:6,apply_plann:6,apt:8,ar:[0,1,2,6,7,9,10,11,13,14,16,18,24,26,30],architectur:[4,19,26,28],arg:[1,4,6,24],args_later:4,args_longitudin:4,args_object:4,argument:[0,1,2,4,5,6,7,9,14],arrai:14,arrang:11,articl:18,asound:8,aspect:11,assert:24,associ:14,astyp:24,async:26,asynchron:26,attach:7,attent:26,attitud:1,aug:14,aug_cfg:7,augmant:7,augment:[7,14],author:[18,19,23],auto:[0,2,4,10,14,15,16,19,20,28,29],auto_pilot:14,auto_reset:[18,19],auto_run:[10,25],auto_run_cas:[10,25],autoencod:22,autom:[18,19],automat:[0,7,11,14,26],autompcpolici:3,autonom:[5,9,14,16,18,19,27,30],autopidpolici:[3,25],autopilot:[6,28],autorun_config:25,avail:[0,2,14],averag:0,avoid:[5,6,7,27,30],b:4,baci:26,back:5,backbon:20,background:14,base:[1,5,6,7,16],base_carla_env:1,base_carla_polici:5,base_env:1,base_env_manag:[0,2],base_simul:6,basecarlaenv:[2,3],basecarlapolici:3,baseenv:1,baseenvinfo:1,baseenvmanag:[0,2,13],baseenvtimestep:1,basesimul:3,bash:26,basic:[7,13,14,16,24,29],basic_plann:7,basicplann:[3,14],batch:[5,13],batch_siz:[18,19,24],becaus:[8,26],been:19,befor:[6,14,18,26],begin:30,beginn:16,behavior:[1,7,10,13,18,19,26],behavior_plann:7,behaviorplann:[3,14],being:1,below:11,benchmark:[0,1,2,10,12,16,19,23,24,28,30],benchmark_dataset_sav:0,benchmark_evalu:2,benchmarkdatasav:19,benchmarkdatasetsav:[3,11],benchmarkenvwrapp:[3,9],benckmark:20,besid:7,best:2,best_ckpt:19,beta1:18,beta2:18,better:[19,28],between:[4,14,15,26],bev:[4,6,14,21,23,25],bev_speed_model:24,bevspeedconvencod:3,bin:[8,26],bird:[4,5,6,14,15,18,23,24,25,30],birdview:[4,14,24,25,30],block:14,booktitl:[19,23],bool:[1,2,6,7,14],both:[4,16,23,28],bradi:23,brake:[4,11,19,23,24,28],branch:19,buffer:[7,24,27,30],build:[2,6,9,14,15,24,28,30],built:[10,26],c:[4,8,17],cach:0,calcul:[1,4,5,7,25,27],call:[2,5,6,13,14,15,16,25,26],callabl:[0,2],camera1_nam:11,camera2_nam:11,camera:[6,7,14,15,18,19,20,25,28],can:[0,1,2,5,6,7,9,10,11,13,14,15,16,17,18,19,20,22,24,25,26,27,28,30],canva:[1,7,15],captur:[19,20],car:7,card:8,carl:14,carla:[0,1,2,5,6,7,9,10,11,12,16,18,19,23,24,25,28,29,30],carla_099:17,carla_0:17,carla_ag:7,carla_benchmark_collector:0,carla_env:[10,26],carla_host:[18,19,25,26,28,30],carla_port:[18,19,25,26,28,30],carla_timeout:1,carlabenchmarkcollector:[3,9,19],carlabenchmarkevalu:[3,9,24],carladataprovid:6,carlaenvwrapp:3,carlascenariosimul:[1,3],carlasim:26,carlasimul:[1,3,14],carlaue4:[17,20,26,28],casezoo:[12,16,23,29],categori:[11,22],caus:6,cautiou:7,cd:[17,20,25,30],ce:26,certain:[7,10,19,28],certainli:1,cfg:[0,1,2,5,6,7,24],chang:[0,1,4,5,6,7,14,16,18,19,23,24,25,26,28,30],changelanetown04:[9,20],channel:[4,6,14,18,24],chaotic:26,character:22,characterist:10,charactorist:10,cheat:21,check:[1,6,7,9,14,15,16,17,24,25],checkout:17,checkpoint:[18,28],chen2019lbc:23,chen:[18,23],choos:[1,9,14,18],cict:18,cict_datasets_train:18,cict_demo:18,cict_ev:18,cict_eval_gan:18,cict_eval_traj:18,cict_gan:18,cict_test:18,cict_train_gan:18,cict_train_traj:18,cict_traj:18,cil:[19,28],cilr:[5,9,19,28],cilrs_config:19,cilrs_data_collect:[19,28],cilrs_datasets_train:19,cilrs_datasets_v:19,cilrs_ev:[19,28],cilrs_test:28,cilrs_train:[19,28],cilrs_val:19,cilrspolici:3,cils_datasets_train:19,ckpt:2,ckpt_path:[19,28,30],clean:[6,7],clean_up:[6,7,26],clear:[0,6,7,14],clear_up:7,client:[4,6,26],clone:[17,19],close:[0,1,2,18,24,28],closer:16,cloudi:14,cnn:19,code:[4,17,19,28,30],codevilla2019explor:19,codevilla:19,col_is_failur:[18,19],col_threshold:[7,14],collat:5,collect:[0,5,11,13,16,18,20,24,27,29,30],collect_data:[18,20],collect_mod:[5,13],collector:[0,2,9,13,18,19,24,27,28],collet:18,collid:[6,7],collis:[6,7,10,14,27],collisionsensor:3,com:[17,22],combin:[4,16],come:[6,10,14],command:[0,8,11,14,17,19,25,26,28],command_index:0,common:[6,10,12,17,18,26],commonli:[9,19,27],commun:26,compar:[2,10],compil:17,complet:[16,30],complex:[13,16],compon:11,comput:[1,4,13,14,19],compute_reward:1,concat:[4,19,24],concept:[10,16,29],concern:30,conda:8,condit:[9,21,28],conf:8,confer:[19,23],config:[0,1,2,5,6,7,10,14,15,18,19,20,25,26,28,30],config_fil:10,configuarion:10,configur:[1,6,10,11,12,16,18,19,28],congratul:28,connect:[17,25,26],conplet:22,consist:[7,9,11,14,24,26],contain:[1,4,5,6,7,9,11,13,14,16,26,27,30],content:[11,14,30],context:[18,19],continu:[21,24],control:[1,4,5,6,10,13,14,16,18,19,23,24,26,27,28,30],control_weight:19,conv:[4,24],conveni:[9,10],convert:[1,6],convolut:4,coordin:[4,14],copi:[20,24],core:[0,1,2,4,5,6,7,10,13,16,24,29,30],corl:23,correct:5,correctli:13,correl:7,correspond:[4,17],cost:28,could:[1,6,8],count:[0,6,7],cpu:30,creat:[1,6,7,10,11,14,15,16,28,29],criteria:[1,6,10],crop:20,cross:7,csv:2,cuda:[19,30],cudnn:19,cur_collector_envstep:24,curcumst:28,current:[1,2,4,6,7,10,14,24,26,30],current_devic:4,current_loc:4,current_ori:4,current_spe:4,current_vec:4,currrent:14,custom:[11,14,15,16,19,24,28],cutin:10,cutin_1:10,cvf:19,d:4,dai:20,data:[1,3,4,5,6,7,9,10,12,13,14,15,16,17,20,25,27,28],data_dict:7,data_id:5,dataset:[0,9,10,12,16,18,19,20,27,29],dataset_dir:20,dataset_metadata:11,dataset_metainfo:0,dataset_nam:11,dataset_path:18,datasets_train:19,ddpg:24,deal:17,debug:[7,14],decis:[7,10,16,27],decod:4,decompos:[16,27],deep:[10,16,27],deepcopi:24,deeper:19,def:24,defalut:26,default_experi:2,defin:[0,1,2,5,7,9,10,11,13,14,16,24,27,30],defini:13,definit:[1,27],delet:[1,6,24],deliv:1,delta_second:14,demo:[10,18,19,20,24,25,28,30],depend:24,deploi:[9,16],deploy:[1,30],depth:14,derect:6,deriv:1,descret:24,describ:11,descript:[2,6,16,25],design:[5,10,13,16,25],desir:11,dest:18,destin:[7,18],destroi:[6,7,26],detail:[1,9,10,14,16,18,20,24,30],detailli:22,detect:7,detector:7,dev:8,develop:[16,17],deviat:4,devic:4,devid:10,di:[0,1,2,4,5,9,10,11,13,14,15,18,19,20,22,23,24,25,26,27,28,29],dian:23,dict:[0,1,2,4,5,6,7,13,14,15,18,19,25,26,28,30],dictanc:[6,14],dictionari:[4,14],differ:[2,5,7,10,11,13,14,24,27,28],differenti:4,difficulti:[10,16],dimens:4,ding:[0,1,2],dir:20,dir_path:[18,19,28],direct:[1,7,14],directli:[7,8,13,30],directori:[11,17,30],disabl:14,disable_two_wheel:[6,14,18,19],discreteenvwrapp:24,discrimin:18,displai:27,dist:17,distanc:[6,7,14],distribut:17,dive:16,divid:11,doc:[9,14,16,24,26,30],docker2:26,docker:[16,29],dockerhub:26,document:[14,22],doe:24,doi:18,done:[1,7,13],dongkun:18,dosovitskii:19,down_channel:18,down_dropout:18,down_norm:18,download:[16,20],dqn:[9,24,30],dqn_eval:30,dqn_test:30,dqn_train:30,draw:14,drive:[0,4,5,9,10,11,13,14,15,18,19,21,22,23,24,25,26,27,28,29],drive_len:4,dropout:18,dt:[4,18],dure:[6,7,10,13,14],dut:6,e:[13,17],each:[0,1,2,4,5,6,7,9,10,11,13,14,18,24,26,27],eas:16,easi:30,easier:26,easili:[7,10,11,16,26],easy_instal:17,ect:6,eder:19,edu:8,effect:[6,11],effici:16,egg:17,ego:[4,7,10],ego_pos:4,elem:0,element:[7,14,26],els:27,embed:[4,22,24],embedding_s:4,enabl:28,enable_field:5,encod:[4,7,20,24],encount:6,end:[1,6,7,9,10,11,14,19,21,25],end_dist:6,end_episod:18,end_idx:6,end_loc:7,end_scenario:6,end_timeout:6,engin:[1,2,5,9,13,16,17,22,24,27,29],entangl:26,entir:[6,9,27],entri:[23,24,28,30],env:[0,2,3,5,8,10,16,18,19,20,25,27,30],env_cfg:[10,26],env_id:13,env_manag:[0,2,13],env_num:[18,19,20,30],env_param:0,env_util:7,envalu:2,envirion:1,environ:[0,1,2,5,6,7,8,9,10,13,15,16,18,20,23,24,27,28,30],envmanag:[0,2,5,9,13],envstep:[2,24],envwrapp:24,episod:[0,1,2,5,6,9,10,12,14,18,28],episode_00000:11,episode_00001:11,episode_count:0,episode_metadata:11,episode_per_suit:2,episodes_data:0,episodes_per_suit:18,epoch:19,equal:[5,13],error:[6,17],essenti:[6,10],establish:[6,10,26],et:9,etc:[8,13,25,27],eu:17,eval:[3,5,9,10,13,18,19,20,24,27,30],eval_config:[18,30],eval_freq:19,eval_mod:13,evalu:[1,2,12,13,16,17,27,29],evalut:20,even:[10,16,26],event:[1,7],everi:[0,2,7,15],everyth:[17,25],eviron:[7,10],ex:[17,20],exactli:10,exampl:[5,9,10,13,14,15,24,26],exclud:24,execut:4,exist:[6,10,16],exp:18,exp_nam:[2,19,30],expect:14,experi:[2,24],expert:[5,28],explain:[11,14,22],explor:19,extend:20,extern:[13,26],ey:[4,5,6,14,15,23,24,25,30],fail:[1,10],failu:10,failur:[1,2,10],fals:[7,14,18,19],faq:[16,17],far:20,farthest:7,fast:26,featur:[4,10,22],feel:16,felip:19,file:[0,1,2,7,8,9,10,11,15,16,24,25,26,28],final_channel:18,find:[0,2,6,8,9,14,26],finish:[10,28],fintun:19,first:[1,2,16,17,18,20,25,28],fix:[8,10,11,26],flexibl:16,float32:24,flow:27,folder:[0,8,11,15,18,20,28],follow:[1,4,5,8,10,11,13,14,16,17,19,20,22,25,26,27,30],follw:11,forg:8,form:[0,2,5,10,11,26,27],format:[1,11,28],former:[10,24],forward:[0,2,4,5,13,14],forward_vector:[11,14],found:[8,15,19,28,30],fov:[14,19],fp:[4,14,28],frac:4,frame:[0,1,4,6,11,14,15,20],frame_skip:[15,30],framework:27,free:[6,16,21,22,26],freeli:11,freez:20,frequenc:2,frequent:[2,4,6],from:[0,1,2,4,5,6,7,8,9,10,11,13,14,17,21,24,26,27,30],front:[7,19,20,28],front_rgb:14,full:[9,19],fulltown01:[9,18,19],fulltown02:[9,30],fulltown04:9,fulltown:20,futur:4,g:[17,28],gaidon:19,gail:27,gan:18,gan_ckpt_path:18,gan_loss_funct:18,gauss:4,gaussian:4,geforc:28,gener:[0,4,6,12,14,18,22,24,26,27],geo:8,geos_c:8,get:[0,1,2,4,5,6,7,8,10,14,15,16,19,22,23,24,27,28,30],get_criteria:6,get_incoming_waypoint_and_direct:7,get_inform:6,get_navig:6,get_observ:1,get_sensor_data:6,get_sensors_data:7,get_stat:6,get_train_sampl:5,get_waypoints_list:7,gif:[1,7,15,30],git:17,github:17,give:19,given:4,global:[7,14,26],go:[14,20],goal:[1,6],gohlk:8,gpu1060:30,gpu:[17,18,20,26],gpu_id:26,graph:13,greatli:16,green:14,ground:23,groundtruth:18,guid:[12,26],guidanc:26,guidenc:16,gym:[1,10,16,24,27],gz:[17,20],h:[4,14],ha:[1,5,7,14,19,26],half:4,hand:[9,16,26],handl:[5,10,13],handler:7,happen:[1,6,8],hard:[10,14],hardwar:[17,28],have:[0,2,5,7,8,11,13,14,16,17,24,27,28,30],hawk:9,head:[20,24],help:[17,27],here:[9,14,15,28,30],hero:[1,4,6,7,10,14,27],hero_play:[1,6],hero_vehicl:7,hidden:4,hidden_dim:[4,18],hidden_dim_list:4,hierach:18,hierarch:18,high:16,histor:17,histori:7,home:16,horizon:4,host:[1,6,10,17,25,26,28,30],hour:[20,28,30],how:[15,20,28,30],howev:10,http:[8,17,20],i7:[28,30],i:[13,17,24],icra:19,id:[5,24],ieee:[18,19],ignor:[1,8,9,30],il:[9,10,18,26,28],illustr:28,imag:[4,5,6,7,11,12,14,15,19,23,24,25,26,27,30],img_height:18,img_step:18,img_width:18,imit:[9,16,18,23,27,29],immedi:27,implement:[19,22,24],implicit:[9,21],impuls:7,in_channel:4,includ:[6,10,14,15,16,19,23,27,28,30],inclut:6,incom:[1,7],index:[0,6,11,25],indic:16,individu:[10,26],indivisu:10,industri:16,info:[1,6,7,14,15],infom:0,inform:[1,4,6,7,9,11,12,13,15,19,20,24,27,30],inherit:1,inheritor:28,init:[1,5,6,7,10,13,14],initi:[1,7],initla:7,inproceed:[19,23],input:[4,5,6,7,13,16,18,19,22,23,28,30],input_dim:18,instal:[16,26],instanc:[0,1,2,4,6,10,13,14,16,20,24],instance_nam:2,instant:27,instantan:7,instruct:[10,14,25],integr:[4,16],intel:30,intellig:[10,16],intens:14,intent:21,interac:10,interact:[0,1,2,5,14,27],interfac:[0,1,2,4,5,6,7,10,13,14,16,24],intern:[13,16,19,26],interv:26,intial:6,introduc:[10,19],invok:4,involv:27,ip:[1,17,18,20,26],irl:27,is_failur:1,is_junct:14,is_success:1,isinst:24,item:[15,24],iter:[2,27],its:[0,5,6,7,9,11,14,17,26,28],itself:10,j:9,jingk:18,journal:18,jpeg:8,json:[0,10,11,25],judg:[2,7],judgement:[1,27],junction:[6,14],just:[8,13,26,28],k_d:4,k_i:4,k_p:4,keep:[7,13],kei:[0,5,10,11,14,15,25,26],kernel:4,kernel_s:[4,18],kind:[5,7,9,13,16,24,27],kl:4,km:[4,14],koltun:[19,23],kr:23,kwarg:[1,4,6,24],l1:[18,19],l:19,label:[13,19,20],lane:[7,9,14,19,20,24,27],larg:30,last:[1,7,11,16,28],latent:[4,21],latent_dim:4,later:4,laterli:27,latter:[10,24],layer:4,lbc:[5,23],lbc_bev_ev:23,lbc_bev_test:23,lbc_image_ev:23,lbc_image_test:23,lbcbirdviewpolici:3,lbcimagepolici:3,leanr:20,learn:[1,5,9,10,13,16,17,18,27,29],learn_mod:13,learner:[13,24,27],learning_r:18,left:14,legal:1,len:24,len_thresh:7,length:[1,4,7],letter:18,level:4,lfd:8,li:18,libcarla:[6,7],libgeo:8,libpng16:8,librari:8,licens:16,lidar:[6,14,18],lidar_nam:11,light:[1,6,7,10,14,15,24,30],lightweight:16,like:[7,9,10,11,13,25,28],limit:[7,14,19],line:[7,8,14,17],lingang:16,link:[6,8,26],linux:17,list:[0,4,5,6,7,9,14,15,20,23,26],literatur:[9,10,16],lmdb:[0,11],load:[1,6,11,28],load_state_dict:13,local:[6,7,14,17],localhost:[1,6,18,19,25,26,28,30],locat:[4,6,7,10,11,14],log:[1,4,8,20,30],log_dir:20,logdir:30,logger:2,logic:[10,13],logvar:4,longitudin:4,look:[11,17],loop:[18,24,28],lopez:19,loss:[4,5,13,18,19],loss_funct:4,low:[4,16],lower_fov:[14,18],lr:19,lra:18,lsof:17,m:19,machin:28,mai:[1,5,6,7,8,10,11,13,14,17,18,23,25,28,30],main:1,mainli:[2,16,26],make:[0,1,5,6,8,9,10,11,13,14,16,20,26,30],make_dataset_path:0,make_index:0,malici:10,manag:[0,1,2,6,18,19,24,26],mani:20,manual:[8,10,17,26],map:[4,6,9,10,11,13,14,18,19,20,24,25,26,27],matthia:19,max:4,max_brak:4,max_ckpt_save_num:18,max_dist:18,max_retri:[18,19],max_steer:4,max_t:18,max_throttl:4,mean:[4,11,26,30],meanwhil:10,measur:[0,6,12,19],measurements_00000:11,meet:17,memori:[17,30],messag:6,met:14,meta:0,metadata:12,metainfo:0,method:[0,1,2,4,5,6,7,9,10,13,14,15,16,18,19,23,24,25,26,27],metric:12,mid:14,middl:4,miiller:19,mimic:[19,28],min:4,min_dist:[7,14],minim:4,minut:11,mkdir:[17,20],mlp:4,mod_valu:24,mode:[1,5,13,14,26,27],model:[3,5,13,16,18,22,27,29],model_configur:18,model_path:20,model_rl:20,model_supervis:20,model_typ:18,model_wrapp:4,models_town01:20,models_town04:20,modif:6,modifi:[14,16,17,18,19,20,27,28,30],modul:[8,9,13,14,16,24,27],modular:16,monitor:30,more:[7,10,13,14,15,20,22,24,30],most:[6,27],mostli:[10,24,28],mount:6,mpc:[4,5],mpccontrol:3,ms:11,mse:18,mu:4,much:26,multi:[6,9,20,24,26,28],multimod:9,multipl:[20,26],must:[0,2,4,6,18],my_polici:13,mypolici:13,n:[4,17,26],n_episod:[0,2,18,19,28],n_epoch:18,n_pedestrian:[6,14],n_sampl:[13,24],n_vehicl:[6,14],name:[2,6,7,10,11,14,18,19,25,28],namedtupl:1,natur:13,navig:[1,6,7,10,12,19,20,25,27,28,30],ndarrai:[1,14],nearbi:[7,14],necessari:11,need:[1,7,8,10,14,16,17,18,20,23,25,26,28,30],net:24,network:[4,13,18,20,27],neural:[13,27],new_data:24,newest:7,next:[6,7,14,26,28],nice:30,nn:[5,13,27],no_rend:14,node:[7,14],node_forward:14,nois:[4,5,18,19],noise_arg:4,noise_kwarg:4,noise_len:4,noise_rang:4,noise_typ:4,none:[0,1,2,4,5,6,7,14,15,18,19,24],noon:14,norm:18,normal:28,note:[0,1,2,17,18,28,30],now:[6,17,28],np:[1,14,24],npc:[6,9,10,14],npy:[19,28],npy_prefix:18,num:[0,7,9,14,18,26,28,30],num_branch:[18,19],num_sampl:4,number:[4,7,9,11,15,18,26,30],number_of_loading_work:18,nvidia:[17,26],o:19,ob:[1,10,11,13,14,15,18,19,24,25],obei:10,oberv:14,object:[4,11],obs_cfg:[0,7],obs_out:24,obs_shap:4,observ:[0,1,5,6,7,11,13,14,15,22,24,25,27],obtain:[1,7,18],occupi:[24,26],od:9,off:[1,6,14,24,30],off_road:6,offici:17,offscreen:20,often:8,ok:17,old:[6,8],ollid:1,onc:[0,2,6,7,9,28],one:[1,2,4,6,7,8,9,11,13,14,15,20,24,26],ones:[7,26],onli:[0,1,2,5,10,14,16,26],onlin:15,onto:[4,15],open:16,opendilab:[16,17,20,26],opengl:20,oper:[5,10,27],option:[0,1,2,4,5,6,7,13,14,15,17,26],order:[1,9,10,11,13,14,27],org:20,organ:11,orient:6,os:10,oserror:8,other:[0,1,5,6,7,9,10,12,13,14,16,19,24,26,27,30],otherwis:20,our:22,out:[1,26],out_dim:18,output:[4,5,13,15,16,19,24,25,27,28,30],overtak:7,overview:12,own:[13,14,17,24,26],p:26,pack:17,packag:17,pad:18,page:[16,17,18,19,20,26],pai:26,paint:[7,14,15],pair:7,parallel:[9,13,24],param:[0,1,2,4,9],paramet:[1,2,6,10,26,27,30],paremet:26,parent_actor:7,pars:[9,10],parse_routes_fil:10,parse_scenario_configur:10,part:[16,25,27,28,30],partit:11,pass:[0,4,9,10,13],path:[0,6,8,15,18,20,28,30],path_to_your_ckpt:28,path_to_your_dataset:28,path_to_your_train_dataset:28,path_to_your_train_preload:28,path_to_your_val_dataset:28,path_to_your_val_preload:28,pcm:8,pe:2,pedestrian:[6,9,11,14,24],pend:23,per:30,percept:27,perform:[2,4,10,12,20,23,24,28],period:6,pez:19,philipp:23,physic:26,pictur:30,pid:[4,5,13,23],pilot:14,pip:[8,17],pipelin:[19,20],pixel_loss_funct:18,pixel_loss_weight:18,pixels_ahead_vehicl:14,pixels_per_met:[14,25],placement:18,plan:[16,27],planner:[6,7,12,18,19,27],planner_dict:14,platform:[1,16],player:6,pleas:[17,24,26,28],plug:8,png:[8,11],point:[6,10],points_per_second:[14,18],polici:[0,1,2,3,9,10,12,16,18,19,20,22,23,24,27,28,29,30],policy_config:18,policy_kwarg:[0,2],polymorph:[13,16],port:[1,6,10,17,18,20,25,26,28,30],posit:[7,10,14,18,19,25],position_rang:14,possess:10,possibl:26,post:[0,18],post_process_fn:0,postpocess:11,postprocess:5,potenti:18,power:13,ppo:24,pre:[18,19,20,30],pre_train:18,pred_len:18,pred_t:18,predict:[4,18,19,23,28],prefix:18,preload:[19,28],preload_model_alia:18,preload_model_batch:18,preload_model_checkpoint:18,prepar:20,prerequisit:[16,29],pretrain:[19,23],print:[1,7,14],priority_info:24,probabl:30,problem:17,procedur:[13,27,28],proceed:[14,19],process:[0,11,16,18,19,23,24,28,29,30],process_transit:5,promot:27,properti:[0,1,2,6,13],proport:4,propreg:5,prosedur:11,protenti:18,provid:[0,1,2,5,6,7,9,11,14,16,19,20,23,24,25,26,28],pth:[18,19,20],publish:8,pull:26,push:24,put:[13,15,20],py3:17,py:[10,18,19,20,23,24,25,28,30],pyenv:8,pypi:17,python:[8,10,14,16,18,20,23,25,26,28,30],pythonapi:17,pythonlib:8,pytorch:[16,17],question:17,queue:[2,7],quick:[16,19,26],quickli:[9,10,16,25,29],r:30,rain:14,raini:14,rais:[10,17],ran:[6,14],ran_light:6,random:[1,6,11,14],randomli:[1,10,14],rang:[4,7,14,18,24,26],rate:[2,9,10,19,24,28],rather:2,reach:[2,4],ready_ob:13,real:[4,10,14,16,28],real_brak:11,real_steer:11,real_throttl:11,realiz:[13,16],reason:26,receiv:7,recommend:[16,24,26,28],reconstruct:4,record:[1,6,7,14,15,17],red:[6,7,14],redesign:10,reduc:16,refer:[6,10,12,22,24,28],reflect:10,regardless:26,regist:7,reinforc:[1,5,13,16,17,27,29],reinstal:8,relat:[1,5,6,17,22],releas:[1,7,17,26],remain:[6,14],remov:[7,18],render:[1,2,7,14,15,27,30],renfirc:16,reparameter:4,repeat:[4,10,24],replac:6,replai:[24,27,30],replay_buff:24,repositori:17,repres:[9,10,11,27],represent:[4,22],requir:[7,17,24],research:28,reset:[0,1,2,5,6,9,10,13,24,25,26],reset_param:2,resnet:19,resolut:[14,18,19],resourc:1,respect:1,result:[1,2,13,14,15,24],resum:19,retriv:1,review:27,reward:[1,2,10,15,25,27],rgb:[5,14,15,18,19,20,23,25,28],rgb_arrai:1,right:14,rl:[9,10,16,17,20,22,24,26,29],road:[1,6,7,9,10,14,15,16,24],roadopt:7,robot:[18,19,23],roll:27,rong:18,root:20,root_dir:[19,28],rotat:[14,18,19,25],rotation_frequ:[14,18],rotation_rang:14,rout:[1,6,7,9,10,14,20,24,25,27,30],route_fil:10,route_pars:10,routepars:10,router:7,rpc:17,rtx:28,run:[0,1,2,4,5,6,7,8,9,12,13,15,16,18,19,23,24,26,27,28,29,30],run_carla:26,run_step:[6,7],run_visu:7,runner:10,runnign:6,runtim:[1,2,26],s3:17,s:[1,7,9,13,14,15,18,26,30],sac:24,safe:7,safeti:7,same:[1,5,6,7,10,11,13,14,20,24,26,28,30],sampl:[0,4,12,13,18,24,25,27,30],santana:19,save:[0,1,2,6,7,10,11,15,20,26,28,30],save_checkpoint:24,save_ckpt_fn:2,save_dir:[0,15,18,20,30],save_episodes_data:0,save_interv:18,saver:0,scalar:[2,4,24],scenario:[1,6,10,16,29],scenario_fil:10,scenario_manag:6,scenario_nam:10,scenario_pars:10,scenariocarlaenv:[3,10,25],scenarioconfigurationpars:10,scenarioenv:10,scenariosimul:10,scneario:6,screen:[1,7,15,25,28,30],scrip:28,script:[17,26],sdl_hint_cuda_devic:20,sdl_videodriv:20,search:7,second:[4,14,18],section:11,see:[17,20,25,28,30],seed:[1,9,11],segment:14,select:[10,14,19,27],self:24,semant:4,send:[6,24,26],sens:[8,14],senser:11,sensor:[1,6,7,12,15,18,25],sensor_data:0,sensor_tick:18,sensor_util:7,sensorhelp:3,sent:1,seper:24,serial:2,serial_evalu:2,serialevalu:[3,24],server:[1,6,14,16,18,19,23,25,28,29,30],set:[1,2,4,5,6,7,9,10,13,14,15,16,18,19,20,22,23,29,30],set_destin:7,set_rout:7,settint:30,setup_sensor:7,setuptool:8,sevar:10,sever:[0,2,5,6,9,10,13,14,26],sh:[17,20,26,28],shanghai:16,shape:4,share:13,shared_memori:[18,19],should:[1,2,5,11,13,20,28],should_ev:[2,24],show:[7,8,9,10,14,15,24,25,26,28,30],show_text:15,shown:[10,11,14,20,25,28],side:[4,26],sigma:4,signal:[1,4,5,6,13,19,24,30],similar:26,similarli:[24,27],simpl:[1,9,10,16,23,24,29],simple_rl:[24,30],simplecarlaenv:[3,11,24,25,26],simpli:[8,14,16,25,30],simplifi:16,simualt:16,simualtor:[6,7,14,26],simul:[0,1,3,7,8,9,10,11,12,15,16,17,18,19,25,26,27],simulator_util:7,simultan:26,singal:[23,28],singl:[1,2,6,9,10,19,20,25,28],single_evalu:2,singlecarlaevalu:3,situat:15,size:[4,14,18,19,24,25,30],skip:15,sky:20,sl:20,slave:8,small:30,so:[1,6,7,10,14,15,19,20,26,27],soft:14,solv:8,some:[1,5,6,7,9,10,13,14,15,17,19,20,26,27],someth:20,soon:22,sound:8,sourc:[0,1,2,4,5,6,7],sourec:16,space:[1,4,24],spawn:[11,14,18,19],special:[0,5],specif:[1,5,10],specifi:[8,10,14,24,26],speed:[4,6,7,11,14,19,21,28,30],speed_factor:18,speed_limit:14,speed_thresh:7,speed_weight:19,squeez:24,srunner:10,stage:[18,20,23],standard:[0,1,4,5,9,10,14,16,19,24,27,30],stare:5,start:[0,1,5,6,7,9,10,11,14,16,19,20,26,28,29],start_episod:[0,18],start_loc:7,state:[1,4,6,7,14,15],state_dict:13,statu:[1,2,6,10,12,27],steer:[4,5,11,19,23,24,28],steernoisewrapp:3,step:[1,2,4,6,7,11,14,15,19,24,26,28],stop:[2,5,24],storag:28,store:[0,1,2,5,6,7,10,11,13,14,26,27],str:[0,1,2,4,5,6,7],straight:[9,14],straightli:1,straighttown01:20,stride:[4,18],structur:[1,12,19,23],stuck:[1,7],stuck_detector:7,stuck_is_failur:[18,19],stuckdetector:3,sub:[14,24],subprocess:20,succe:[1,2,10],success:[1,2,9,10,19,27],successfulli:[1,17],sucha:6,sudo:8,suggest:[11,14],suit:[0,1,2,9,10,11,13,18,19,20,24,28,30],suitabl:[10,13,27],suite_nam:28,summari:4,summarywrit:2,sunset:14,supervis:[5,13,20],supervised_model:20,supervised_model_path:20,support:[5,14,16,18,19,27,28],suppos:1,sure:[0,6,26],surround:[6,7],sync:[14,26],sync_mod:14,synchron:[7,14,26],system:[6,8,17,26,30],tabl:[23,28],tag:[11,14,15,17],tailgat:7,take:[4,6,7,14,15,19,20,23,24,28,30],taken:7,tar:[17,20],target:[1,4,5,6,7,9,14,16,23,24,29],target_forward:14,target_loc:4,target_spe:[4,18,19],task:[1,16,22],tb_logger:2,tcp:[1,6,26],td3:24,teach:23,techic:22,tensor:[4,24],tensorboard:[2,30],tensorboardx:2,term:4,termin:30,test:[9,10,13,16,18,20,24,28,29],test_config:30,text:15,than:[2,14],thecarla:6,thei:[9,10,14,24],them:[0,2,7,8,11,24,25],thi:[1,2,6,7,8,10,13,14,15,16,17,19,20,24,25,26,28],three:[9,10,13,14,28],thresh:7,threshold:[7,14],throttl:[4,11,19,23,24,28],through:[4,14,25,26],throughout:14,thu:27,thw:4,tick:[1,5,6,7,11,14,26],time:[1,2,4,6,7,13,14,20,24,26,28],timeout:[6,7],timestamp:[11,14],timestep:24,titl:[18,19,23],tl_di:[11,14],tl_state:[11,14],tm:[26,30],tm_port:[1,6,26],togeth:[6,10,13,18,19,23,27,28,30],too:30,tool:[10,16],toolkit:10,top_rgb:14,torch:[4,24],total:[1,9,11,14,18],total_diat:6,total_light:14,total_lights_ran:14,town01:[9,14,19],town02:[9,19,20],town03:10,town04:10,town05:10,town1:9,town2:[9,20],town:[6,9,10,11,14,18,19,20,30],town_nam:6,trace:[6,7],track:[7,14],traffic:[1,6,7,10,14,15,24,26,30],trafficlighthelp:3,train:[2,9,10,13,16,17,26,27,29],train_config:[18,19,30],train_data:[13,24],train_dataset_nam:18,train_host_port:20,train_it:[2,24],train_rl:20,train_sl:20,traj_ckpt_path:18,traj_model:18,trajectori:[0,4,21],transfer:22,transform:[19,28],transform_ob:[18,19],tree:1,trick:4,trigger:[1,10],truth:23,tune:30,tupl:[1,2,4,7],turn:[9,14],tutori:[16,19,28],two:[4,11,14,15,18,20,23,24,25,26,27],txt:[0,9],type:[4,8,10,11,14,15,18,19,25,30],ubuntu:[8,17,30],uci:8,uhl:23,uncontrol:10,under:[13,16,17,19,28],unifi:[11,14,16],uniform:4,union:0,unless:2,unpack_birdview:24,unpair:18,unreach:7,unsuccessfulli:1,up:[4,6,7,9,10,14,17,19,28],up_channel:18,up_dropout:18,up_norm:18,updat:[5,6,7,16,24,26,27],update_per_collect:24,upper_fov:[14,18],urban:[9,21],us:[0,1,2,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29],usag:[10,16,24],user:[0,2,5,7,9,10,11,13,14,15,16,17,26],usual:[8,13,26,30],util:[1,3],v100:20,v1:[9,18,19,30],v2:[9,30],v3:[9,18],v4:9,vae:4,vae_model:4,val:[19,28],val_host_port:20,valid:28,valu:[1,2,4,5,7,14,24],vanilla:4,vanillava:3,vari:[27,30],variat:[4,22],varieti:16,variou:[10,13],vector:[4,14],vehicl:[1,4,5,6,7,9,10,11,14,16,19,24,26,27,28],vehiclepidcontrol:3,veloc:18,velocity_loss_weight:18,verbos:[14,18,19],veri:[10,30],version:[6,8,9,17,27],vi:18,via:[16,17,18,19,29],video:[1,7,15,30],view:[4,5,6,14,15,18,23,24,25,30],vision:19,visual:[1,2,3,12,16,18,19,23,24,27,28,29,30],visualizas:15,vladlen:[19,23],volum:18,w:4,wai:[5,7,13,16,20,24,30],wait:26,walker:[6,7,10,14],wang:18,want:[17,20,24,25,28,30],wapoint:7,watch:15,waypoint:[1,4,5,6,7,9,11,14,16,23,29],waypoint_list:14,waypoint_num:[7,14,18,19],we:[9,10,11,14,15,17,18,19,20,24,25,26,28,30],weather:[6,9,10,11,14,26],weight:[4,18,19,20,23,30],weight_decai:18,well:[5,6,10,16,19,20,24,25,27],west:17,wet:14,wget:[17,20],what:[10,13],whatev:25,wheel:[8,14],when:[0,1,2,5,6,7,10,13,14,15,17,25,26,28,30],whether:[1,2,6,7,14,15,17],which:[1,4,5,6,9,10,13,14,18,19,22,23,25,26,27],whose:26,wide:9,window:[7,8,17],winerror:8,within:[4,7,26],without:[0,2,10,16],work:[1,5,13,24,25,30],worker:13,world:[6,7,14,17,20,26,27,28],wrap:[1,4,9],wrapper:[1,4,18,19,24],write:0,writer:2,written:[24,30],writter:[2,7],wrong:[1,6],wrong_direct:6,www:8,x86_64:17,x:[4,14],xiao:9,xiong:18,xml:[10,25],xvf:20,xvzf:17,xxx:10,xxx_mode:[5,13],y:[4,9,14],yaw:4,year:[18,19,23],yellow:14,you:[2,6,8,9,10,14,15,16,17,18,19,20,23,24,25,26,28,30],your:[8,14,19,20,23,24,25,26,28,30],yourself:8,yue:18,yuehua:18,yunkai:18,z:4,zexi:18,zhang:18,zhou:23,zoo:16},titles:["data","envs","eval","API Doc","models","policy","simulators","utils","FAQ","Benchmark Evaluation","Casezoo Evaluation","Datasets","Features","Policy Features","Simulator Features","Visualization","DI-drive Documentation","Installation","from Continuous Intention to Continuous Trajectory","Conditional Imitation Learning","End-to-End Model-Free Reinforcement Learning for Urban Driving using Implicit Affordances","Model Zoo","Latent Reinforcement Learning","Learning by Cheating","BeV Speed End-to-end Reinforcement Learning","Auto policy running and visualization","Carla tutorial","Core Concepts and Processes","Simple Imitation Learning","Tutorial","Simple Reinforcement Learning"],titleterms:{"import":8,No:8,With:20,afford:20,alsa:8,api:[3,17],auto:25,autompcpolici:5,autopidpolici:5,basecarlaenv:1,basecarlapolici:5,basesimul:6,basic:26,basicplann:7,behaviorplann:7,benchmark:[9,11,18,20],benchmarkdatasetsav:0,benchmarkenvwrapp:1,bev:24,bevspeedconvencod:4,can:8,carla:[8,14,17,20,26],carlabenchmarkcollector:0,carlabenchmarkevalu:2,carlaenvwrapp:1,carlascenariosimul:6,carlasimul:6,casezoo:[10,25],cheat:23,cilrspolici:5,collect:[19,23,28],collisionsensor:7,common:14,concept:[26,27],condit:19,configur:14,confxxx:8,content:16,continu:18,core:27,cost:20,creat:26,data:[0,11,18,19],dataset:[11,23,28],di:[16,17,30],displai:20,doc:3,docker:26,document:16,download:17,drive:[16,17,20,30],easy_instal:8,egg:8,end:[20,24],engin:30,env:[1,24],episod:11,error:8,eval:2,evalu:[9,10,18,19,20,23,24,28,30],faq:8,featur:[12,13,14,16],free:20,from:18,gener:11,get:17,guid:10,imag:9,imit:[19,21,28],implicit:20,inform:14,input:24,instal:[8,17],intent:18,latent:22,lbcbirdviewpolici:5,lbcimagepolici:5,learn:[19,20,21,22,23,24,28,30],lib:8,libjpeg:8,libpng:8,main:16,measur:11,metadata:11,method:21,metric:9,model:[4,19,20,21,23,24,28,30],mpccontrol:4,navig:14,nn:24,other:[11,21],overview:[10,14],perform:9,planner:14,polici:[5,13,25],prepar:18,prerequisit:[17,30],problem:8,process:27,python:17,q1:8,q2:8,q3:8,q4:8,q5:8,quickli:26,refer:9,reinforc:[20,21,22,24,30],result:[18,20],rl:30,run:[10,14,17,20,25],sampl:9,scenario:25,scenariocarlaenv:1,sensor:[11,14],sensorhelp:7,serialevalu:2,server:[17,20,26],set:26,shape:8,simpl:[28,30],simplecarlaenv:1,simul:[6,14],singlecarlaevalu:2,speed:24,start:25,statu:14,steernoisewrapp:4,structur:11,stuckdetector:7,tabl:16,target:25,test:[23,30],town01:20,town04:20,trafficlighthelp:7,train:[18,19,20,22,23,24,28,30],trajectori:18,tutori:[26,29],urban:20,us:[20,30],util:7,vae:22,vanillava:4,vehiclepidcontrol:4,via:26,visual:[7,15,25],waypoint:25,without:20,zoo:21}})